from ch_milestones.recording.lerobot_format import TASK_MESSAGE_FEATURE, features


def test_lerobot_features_store_serialized_task_message():
    dataset_features = features((256, 288, 3), use_videos=True)

    assert dataset_features[TASK_MESSAGE_FEATURE] == {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    }
