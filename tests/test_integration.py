"""
Integration test script to verify all pipelines work correctly with mock data.
Tests the core functionality and connectivity of the system.
"""

import sys
import logging
from pathlib import Path
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent / "utils"))


def test_all():
    """Run all core tests."""
    logger.info("\n" + "=" * 70)
    logger.info("MLOPS PREDICTIVE MAINTENANCE - Integration Test Suite")
    logger.info("=" * 70 + "\n")

    results = {}

    # Test 1: Settings Manager
    logger.info("Test 1: Settings Manager")
    try:
        from utils.settings_manager import SettingsManager
        manager = SettingsManager()
        settings = manager.load()
        logger.info(f"  ✓ Settings loaded successfully")
        logger.info(f"    Project: {settings.project_name}")
        logger.info(f"    Target variable: {settings.schema.target_variable}")
        results["Settings Manager"] = True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Settings Manager"] = False

    # Test 2: Utility Functions
    logger.info("\nTest 2: Utility Functions")
    try:
        from utils.core_utils import get_logger, ensure_dir, require_columns
        test_logger = get_logger("test_logger", log_dir=None)
        ensure_dir("test_dir")
        df = pd.DataFrame({"col1": [1, 2]})
        require_columns(df, ["col1"])
        logger.info(f"  ✓ All utility functions work")
        results["Utility Functions"] = True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Utility Functions"] = False

    # Test 3: Error Handling
    logger.info("\nTest 3: Error Handling")
    try:
        from utils.pipeline_errors import ConfigurationFault, SchemaValidationFault
        error = ConfigurationFault("Test error")
        error.serialize()
        logger.info(f"  ✓ Error handling works correctly")
        results["Error Handling"] = True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Error Handling"] = False

    # Test 4: Enum types
    logger.info("\nTest 4: Pipeline Enums")
    try:
        from utils.pipeline_enums import UserSegment, OptimizationGoal
        assert UserSegment.POWER_USER.value == "power_user"
        assert OptimizationGoal.BALANCED.value == "balanced"
        logger.info(f"  ✓ Enum types work correctly")
        results["Enum Types"] = True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Enum Types"] = False

    # Test 5: Feature Builders
    logger.info("\nTest 5: Feature Engineering Utilities")
    try:
        from utils.feature_builders import split_by_segment, build_power_user_features, build_guest_features
        df = pd.DataFrame({
            "user_id": ["u1", "u1", "u2", "u3"],
            "session_id": ["s1", "s2", "s3", "s4"],
            "duration_minutes": [15, 45, 10, 5],
            "device_type": ["mobile", "desktop", "mobile", "tablet"],
            "event_date": pd.to_datetime(["2023-10-01", "2023-10-05", "2023-10-02", "2023-10-06"])
        })
        thresholds = {"power_user": 2, "guest": 1}
        power_df, casual_df, guest_df = split_by_segment(df, thresholds)
        logger.info(f"  ✓ Feature builders work correctly")
        logger.info(
            f"    Power users: {len(power_df)} records, Guests: {len(guest_df)} records")
        results["Feature Builders"] = True
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Feature Builders"] = False

    # Test 6: Preprocessing Pipeline (optional, requires mlflow)
    logger.info("\nTest 6: Preprocessing Pipeline (optional)")
    try:
        from pipelines.preprocessing_pipeline import PreprocessingPipeline
        from utils.pipeline_enums import OptimizationGoal
        pipeline = PreprocessingPipeline(goal=OptimizationGoal.BALANCED)
        output_path = pipeline.run()
        assert Path(output_path).exists()
        df = pd.read_parquet(output_path)
        logger.info(f"  ✓ Preprocessing pipeline works")
        logger.info(f"    Output: {len(df)} records")
        results["Preprocessing Pipeline"] = True
    except ImportError as e:
        logger.info(f"  ⊘ Skipped (missing dependency: {str(e).split()[-1]})")
        results["Preprocessing Pipeline"] = None
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Preprocessing Pipeline"] = False

    # Test 7: Feature Pipeline (optional)
    logger.info("\nTest 7: Feature Pipeline (optional)")
    try:
        from pipelines.feature_pipeline import FeaturePipeline
        # Create a test preprocessed data
        test_data = pd.DataFrame({
            "user_id": ["U1", "U1", "U2", "U3"],
            "session_id": ["S1", "S2", "S3", "S4"],
            "event_date": pd.to_datetime(["2023-10-01", "2023-10-05", "2023-10-02", "2023-10-06"]),
            "duration_minutes": [15, 45, 10, 5],
            "device_type": ["mobile", "desktop", "mobile", "tablet"]
        })
        Path("artifacts").mkdir(exist_ok=True)
        test_path = Path("artifacts") / "test_preprocessed.parquet"
        test_data.to_parquet(test_path)

        pipeline = FeaturePipeline()
        feature_paths = pipeline.run(str(test_path))
        logger.info(f"  ✓ Feature pipeline works")
        logger.info(f"    Features generated: {len(feature_paths)} segments")
        results["Feature Pipeline"] = True
    except ImportError as e:
        logger.info(f"  ⊘ Skipped (missing dependency: {str(e).split()[-1]})")
        results["Feature Pipeline"] = None
    except Exception as e:
        logger.error(f"  ✗ Failed: {e}")
        results["Feature Pipeline"] = False

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)

    passed = sum(1 for v in results.values() if v is True)
    skipped = sum(1 for v in results.values() if v is None)
    failed = sum(1 for v in results.values() if v is False)

    for test_name, result in results.items():
        if result is True:
            status = "✓ PASS"
        elif result is None:
            status = "⊘ SKIP"
        else:
            status = "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 70)
    logger.info(
        f"Results: {passed} passed, {skipped} skipped, {failed} failed")
    logger.info("=" * 70 + "\n")

    if failed == 0:
        logger.info(
            "✓ All required tests passed! Repository is working correctly.")
        return 0
    else:
        logger.error(
            f"✗ {failed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = test_all()
    sys.exit(exit_code)
