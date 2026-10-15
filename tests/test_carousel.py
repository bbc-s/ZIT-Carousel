import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch
from tempfile import TemporaryDirectory

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
spec = importlib.util.spec_from_file_location("zit_carousel", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
carousel = sys.modules["zit_carousel.carousel"]
vision = sys.modules["zit_carousel.vision"]


class Clip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        return tokens


class VAE:
    def encode(self, image):
        return image.movedim(-1, 1).clone()

    def decode(self, latent):
        return latent.movedim(1, -1).clone()


class CarouselTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(
            model=object(), clip=Clip(), vae=VAE(), prompt="A person sitting on a bench",
            batch_size=4, seed=21, width=16, height=16, steps=9, cfg=1.0,
            sampler_name="euler", scheduler="simple", denoise=0.4,
            variation_mode="poses", custom_shots="", image=torch.full((1, 16, 16, 3), 0.25),
            reference_description="Curly red hair, green coat, oak bench, soft light",
            vision_model="off",
        )
        self.calls = []
        self.masks = []

        def sample(model, seed, steps, cfg, sampler, scheduler, positive, negative, latent, **kwargs):
            self.calls.append((seed, latent["samples"].clone(), positive, kwargs))
            self.masks.append(latent.get("noise_mask"))
            # Mutating a sampler input must not change the shared reference.
            latent["samples"].add_(0.1)
            return (latent,)

        self.sampler = patch.object(carousel.nodes, "common_ksampler", side_effect=sample)
        self.device = patch.object(carousel.comfy.model_management, "intermediate_device", return_value=torch.device("cpu"))
        self.sampler.start()
        self.device.start()
        self.addCleanup(self.sampler.stop)
        self.addCleanup(self.device.stop)

    def run_node(self, **overrides):
        return carousel.ZITCarousel().generate(**(self.inputs | overrides))

    def test_every_shot_uses_original_reference(self):
        images, report = self.run_node()
        self.assertEqual(tuple(images.shape), (4, 16, 16, 3))
        self.assertEqual([call[0] for call in self.calls], [22, 23, 24])
        for _, latent, prompt, options in self.calls:
            self.assertTrue(torch.equal(latent, torch.full_like(latent, 0.25)))
            self.assertIn("sitting on a bench", prompt)
            self.assertIn("Curly red hair", prompt)
            self.assertEqual(options["denoise"], 0.4)
        self.assertTrue(torch.equal(images[:1], self.inputs["image"]))
        self.assertEqual(len(carousel.json.loads(report)["shots"]), 4)

    def test_generated_reference_is_first_and_shared(self):
        empty = {"samples": torch.zeros(1, 3, 16, 16)}
        with patch.object(carousel.EmptySD3LatentImage, "execute", return_value=(empty,)):
            images, report = self.run_node(image=None)
        self.assertEqual([call[0] for call in self.calls], [21, 22, 23, 24])
        self.assertNotIn("denoise", self.calls[0][3])
        for call in self.calls[1:]:
            self.assertTrue(torch.allclose(call[1], torch.full_like(call[1], 0.1)))
        self.assertEqual(images.shape[0], 4)
        self.assertEqual(carousel.json.loads(report)["shots"][0]["kind"], "generated_reference")

    def test_one_reference_does_not_sample_or_encode_vae(self):
        with patch.object(self.inputs["vae"], "encode", side_effect=AssertionError("unexpected encode")):
            images, _ = self.run_node(batch_size=1)
        self.assertEqual(images.shape[0], 1)
        self.assertEqual(self.calls, [])

    def test_custom_shots_cycle_and_seed_wraps(self):
        self.run_node(seed=(1 << 64) - 1, variation_mode="custom", custom_shots="left view\n\nright view")
        self.assertEqual([call[0] for call in self.calls], [0, 1, 2])
        self.assertTrue(self.calls[0][2].startswith("Photograph: left view."))
        self.assertTrue(self.calls[1][2].startswith("Photograph: right view."))
        self.assertTrue(self.calls[2][2].startswith("Photograph: left view."))

    def test_reference_dimensions_override_widgets(self):
        images, _ = self.run_node(image=torch.zeros(1, 19, 27, 3))
        self.assertEqual(tuple(images.shape), (4, 16, 24, 3))

    def test_invalid_references_fail_before_sampling(self):
        for image in (torch.zeros(2, 16, 16, 3), torch.zeros(1, 16, 16, 4), torch.zeros(1, 4, 4, 3)):
            with self.subTest(shape=image.shape), self.assertRaises(ValueError):
                self.run_node(image=image)
        self.assertEqual(self.calls, [])

    def test_invalid_options_fail_before_sampling(self):
        for options in (dict(batch_size=0), dict(batch_size=65), dict(denoise=0),
                        dict(variation_mode="unknown"), dict(variation_mode="custom", custom_shots="\n")):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_node(**options)
        self.assertEqual(self.calls, [])

    def test_cancellation_is_propagated(self):
        with patch.object(carousel.comfy.model_management, "throw_exception_if_processing_interrupted", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                self.run_node()
        self.assertEqual(self.calls, [])

    def test_automatic_analysis_runs_once_and_replaces_reference_pose_text(self):
        details = {key: f"fixed {key}" for key in vision.DETAIL_FIELDS}
        details["person_box"] = [0.3, 0.2, 0.6, 0.8]
        with patch.object(carousel, "select_vision_model", return_value="local-vlm"), \
             patch.object(carousel, "analyze_reference", return_value=details) as analyze, \
             patch.object(carousel.comfy.model_management, "unload_all_models"):
            images, report = self.run_node(vision_model="auto", reference_description="")
        analyze.assert_called_once()
        self.assertEqual(carousel.json.loads(report)["analysis"], details)
        self.assertEqual(carousel.json.loads(report)["reference_mode"], "person_inpaint")
        for call in self.calls:
            self.assertIn("fixed clothing", call[2])
            self.assertNotIn(self.inputs["prompt"], call[2])
        self.assertTrue(all(mask is not None for mask in self.masks))
        # The mock sampler changes every pixel; compositing must restore the background.
        self.assertTrue(torch.equal(images[1:, :, :2], images[:1, :, :2].expand(3, -1, -1, -1)))
        self.assertGreater(images[1, 8, 7, 0].item(), images[0, 8, 7, 0].item())

    def test_camera_uses_whole_frame_even_with_analysis(self):
        details = {key: f"fixed {key}" for key in vision.DETAIL_FIELDS}
        details["person_box"] = [0.3, 0.2, 0.6, 0.8]
        with patch.object(carousel, "select_vision_model", return_value="local-vlm"), \
             patch.object(carousel, "analyze_reference", return_value=details), \
             patch.object(carousel.comfy.model_management, "unload_all_models"):
            _, report = self.run_node(vision_model="auto", reference_description="", variation_mode="camera")
        self.assertEqual(carousel.json.loads(report)["reference_mode"], "whole_image")
        self.assertEqual(self.masks, [None, None, None])

    def test_malformed_analysis_fails(self):
        details = {key: f"fixed {key}" for key in vision.DETAIL_FIELDS}
        details["person_box"] = [0.1, 0.2, 0.8, 0.9]
        self.assertEqual(vision.parse_analysis(carousel.json.dumps(details)), details)
        for text in ("not json", "```", "{}", carousel.json.dumps(details | {"person_box": [0, 0, 100, 100]})):
            with self.subTest(text=text), self.assertRaises(ValueError):
                vision.parse_analysis(text)

    def test_person_mask_is_finite_and_within_image(self):
        for box in ([0, 0, 1, 1], [0.4, 0.3, 0.5, 0.8]):
            mask = carousel.person_mask(torch.zeros(1, 128, 192, 3), box)
            self.assertEqual(tuple(mask.shape), (1, 128, 192))
            self.assertTrue(torch.isfinite(mask).all())
            self.assertGreaterEqual(mask.min().item(), 0)
            self.assertEqual(mask.max().item(), 1)

    def test_inpaint_cannot_silently_change_camera(self):
        with self.assertRaisesRegex(ValueError, "fixed camera"):
            self.run_node(reference_mode="person_inpaint", variation_mode="mixed")

    def test_local_model_selection_rejects_paths_and_missing_weights(self):
        class LocalNode:
            @classmethod
            def INPUT_TYPES(cls):
                return {"required": {"model_name": (["local-vlm", "incomplete", "../escape"],)}}

        with TemporaryDirectory() as directory:
            root = Path(directory) / "LLM" / "Qwen-VL"
            target = root / "local-vlm"
            target.mkdir(parents=True)
            (target / "config.json").write_text("{}")
            (target / "model.safetensors").touch()
            (root / "incomplete").mkdir()
            with patch.dict(vision.nodes.NODE_CLASS_MAPPINGS, {"AILab_QwenVL": LocalNode}), \
                 patch.object(vision.folder_paths, "models_dir", directory), \
                 patch.object(vision.folder_paths, "folder_names_and_paths", {}):
                self.assertEqual(list(vision.local_vision_models()), ["local-vlm"])
                self.assertEqual(vision.select_vision_model("auto"), "local-vlm")
                with self.assertRaisesRegex(ValueError, "no selected local"):
                    vision.select_vision_model("../escape")

    def test_vision_adapter_uses_reference_and_releases_model(self):
        details = {key: f"fixed {key}" for key in vision.DETAIL_FIELDS}
        details["person_box"] = [0, 0, 1, 1]
        instance = Mock()
        instance.process.return_value = (carousel.json.dumps(details),)
        node = Mock(return_value=instance)
        with patch.dict(vision.nodes.NODE_CLASS_MAPPINGS, {"AILab_QwenVL": node}):
            self.assertEqual(vision.analyze_reference(self.inputs["image"], "local-vlm", 21), details)
        options = instance.process.call_args.kwargs
        self.assertIs(options["image"], self.inputs["image"])
        self.assertFalse(options["keep_model_loaded"])
        self.assertIn("person_box", options["custom_prompt"])


if __name__ == "__main__":
    unittest.main()
