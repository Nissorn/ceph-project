#!/usr/bin/env python3
"""
Smoke test for the segmentation pipeline.
Tests:
1. Dummy Dataloader: Creates fake polygon data, loads dataset, checks tensor shapes
2. Model Forward/Backward Pass: Tests U-Net model with SegmentationLoss
"""

import json
import torch
import numpy as np
from pathlib import Path
import sys

# Add project root to path so we can import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase2b.segmentation_dataset import SegmentationDataset
from src.phase2b.segmentation import build_segmentation_model, SegmentationLoss, POLYGON_CLASSES


def create_dummy_landmarks_data():
    """Create a dummy landmarks_clean.json structure in memory with fake polygon data."""

    # Create a dummy record with polygon data for all three classes
    dummy_record = {
        "image_id": "Dummy_T1",
        "filename": "dummy.jpg",
        "patient_id": "DummyPatient",
        "timepoint": "T1",
        "width": 1729,
        "height": 2048,
        "calibration_pts": [
            [1454.7, 135.86],
            [1461.13, 440.95]
        ],
        "has_calibration": True,
        "keypoints": [],  # Not needed for segmentation test
        "valid_mask": [],
        "has_landmarks": False,
        "polygons": {
            # Create simple square polygons for each class
            "Upper_incisor": [
                [800.0, 800.0],
                [900.0, 800.0],
                [900.0, 900.0],
                [800.0, 900.0]
            ],
            "Labial_bone": [
                [700.0, 700.0],
                [800.0, 700.0],
                [800.0, 800.0],
                [700.0, 800.0]
            ],
            "Palatal_bone": [
                [600.0, 600.0],
                [700.0, 600.0],
                [700.0, 700.0],
                [600.0, 700.0]
            ]
        }
    }

    return [dummy_record]


def test_dummy_dataloader():
    """Test 1: Create dummy data, initialize dataset, check tensor shapes."""
    print("=" * 60)
    print("TEST 1: Dummy Dataloader Test")
    print("=" * 60)

    # Create dummy data
    dummy_records = create_dummy_landmarks_data()

    # Create a temporary directory for dummy images
    dummy_image_dir = project_root / "data" / "raw"
    dummy_image_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image file (we'll create a small black image)
    dummy_image_path = dummy_image_dir / "dummy.jpg"
    if not dummy_image_path.exists():
        # Create a small black image using OpenCV
        import cv2
        dummy_img = np.zeros((2048, 1729, 3), dtype=np.uint8)  # Height, Width, Channels
        cv2.imwrite(str(dummy_image_path), dummy_img)
        print(f"Created dummy image: {dummy_image_path}")

    try:
        # Initialize the segmentation dataset
        dataset = SegmentationDataset(
            records=dummy_records,
            image_dir=str(dummy_image_dir),
            input_size=(512, 512),
            require_polygons=True
        )

        print(f"Dataset length: {len(dataset)}")
        assert len(dataset) == 1, f"Expected 1 record, got {len(dataset)}"

        # Fetch one item
        img_tensor, mask_tensor, meta = dataset[0]

        print(f"Image tensor shape: {img_tensor.shape}")
        print(f"Mask tensor shape: {mask_tensor.shape}")
        print(f"Meta: {meta}")

        # Assert expected shapes
        expected_img_shape = (3, 512, 512)  # C, H, W
        expected_mask_shape = (3, 512, 512)  # C, H, W (one channel per polygon class)

        assert img_tensor.shape == expected_img_shape, \
            f"Expected image shape {expected_img_shape}, got {img_tensor.shape}"
        assert mask_tensor.shape == expected_mask_shape, \
            f"Expected mask shape {expected_mask_shape}, got {mask_tensor.shape}"

        # Check value ranges
        assert img_tensor.min() >= 0.0 and img_tensor.max() <= 1.0, \
            f"Image tensor values should be in [0, 1], got [{img_tensor.min()}, {img_tensor.max()}]"
        assert mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0, \
            f"Mask tensor values should be in [0, 1], got [{mask_tensor.min()}, {mask_tensor.max()}]"
        assert mask_tensor.dtype == torch.float32, \
            f"Mask tensor should be float32, got {mask_tensor.dtype}"

        print("✅ Dummy dataloader test PASSED")
        print(f"   - Image shape: {img_tensor.shape} (expected: {expected_img_shape})")
        print(f"   - Mask shape: {mask_tensor.shape} (expected: {expected_mask_shape})")
        print(f"   - Image range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
        print(f"   - Mask range: [{mask_tensor.min():.3f}, {mask_tensor.max():.3f}]")

        return True

    except Exception as e:
        print(f"❌ Dummy dataloader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up dummy image
        if dummy_image_path.exists():
            dummy_image_path.unlink()
        # Remove dummy directory if empty
        try:
            dummy_image_dir.rmdir()
        except OSError:
            pass  # Directory not empty, ignore


def test_model_forward_backward():
    """Test 2: Model forward and backward pass with dummy data."""
    print("\n" + "=" * 60)
    print("TEST 2: Model Forward & Backward Pass Test")
    print("=" * 60)

    try:
        # Instantiate the U-Net model
        model = build_segmentation_model(
            num_classes=len(POLYGON_CLASSES),
            encoder_name="resnet34",
            pretrained=True  # Use ImageNet pretrained weights
        )
        print(f"Model created: {type(model).__name__}")

        # Instantiate the loss function
        criterion = SegmentationLoss(smooth=1e-6)
        print(f"Loss function created: {type(criterion).__name__}")

        # Create dummy input batch and target masks
        batch_size = 2
        channels = 3
        height, width = 512, 512

        # Input: random noise (simulating image data)
        X = torch.randn(batch_size, channels, height, width, requires_grad=True)
        print(f"Input tensor X shape: {X.shape}")

        # Target: random binary masks (simulating ground truth)
        Y = torch.randint(0, 2, (batch_size, len(POLYGON_CLASSES), height, width)).float()
        print(f"Target tensor Y shape: {Y.shape}")

        # Forward pass
        print("Running forward pass...")
        preds = model(X)
        print(f"Predictions tensor shape: {preds.shape}")

        # Check that predictions shape matches target
        assert preds.shape == Y.shape, \
            f"Predictions shape {preds.shape} != target shape {Y.shape}"

        # Check that predictions are raw logits (can be any real number)
        print(f"Predictions range: [{preds.min():.3f}, {preds.max():.3f}]")

        # Compute loss
        print("Computing loss...")
        loss = criterion(preds, Y)
        print(f"Loss value: {loss.item():.6f}")

        # Check that loss is a scalar tensor
        assert loss.dim() == 0, f"Loss should be scalar, got {loss.dim()} dimensions"
        assert loss.item() >= 0.0, f"Loss should be non-negative, got {loss.item()}"

        # Backward pass
        print("Running backward pass...")
        loss.backward()

        # Check that gradients were computed
        assert X.grad is not None, "Gradients should be computed for input X"
        print(f"Input gradients shape: {X.grad.shape}")
        print(f"Input gradients range: [{X.grad.min():.6f}, {X.grad.max():.6f}]")

        # Check that model parameters have gradients
        total_params = 0
        params_with_grad = 0
        for param in model.parameters():
            total_params += param.numel()
            if param.grad is not None:
                params_with_grad += param.numel()

        print(f"Model parameters: {total_params:,}")
        print(f"Parameters with gradients: {params_with_grad:,}")
        assert params_with_grad > 0, "Some model parameters should have gradients"

        print("✅ Model forward/backward test PASSED")
        print(f"   - Input shape: {X.shape}")
        print(f"   - Output shape: {preds.shape}")
        print(f"   - Loss: {loss.item():.6f}")
        print(f"   - Gradients computed for {params_with_grad:,}/{total_params:,} parameters")

        return True

    except Exception as e:
        print(f"❌ Model forward/backward test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("SEGMENTATION PIPELINE SMOKE TEST")
    print("Testing U-Net segmentation model and dataset components\n")

    # Run tests
    test1_passed = test_dummy_dataloader()
    test2_passed = test_model_forward_backward()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Test 1 (Dummy Dataloader): {'PASSED' if test1_passed else 'FAILED'}")
    print(f"Test 2 (Model Fwd/Bwd):    {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! Segmentation pipeline is ready.")
        return 0
    else:
        print("\n💥 SOME TESTS FAILED! Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())