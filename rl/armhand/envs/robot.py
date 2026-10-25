import math

import numpy as np

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]

ARM_STIFFNESS = np.asarray([600.0, 600.0, 500.0, 400.0, 200.0, 200.0, 200.0], dtype=np.float64)
ARM_DAMPING = np.asarray(
    [
        27.027026473513512,
        27.027026473513512,
        24.672186769721083,
        22.067474708266914,
        9.752538131173853,
        9.147747263670984,
        9.147747263670984,
    ],
    dtype=np.float64,
)
ARM_EFFORT = np.asarray([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0], dtype=np.float64)
ARM_VELOCITY_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

HAND_JOINT_DRIVE = {
    "right_index_MCP_FE": (6.95, 0.28676845, 0.0032, 0.132),
    "right_index_MCP_AA": (13.2, 0.40845109, 0.0032, 0.132),
    "right_index_PIP": (4.76, 0.20394083, 0.00265, 0.07456),
    "right_index_DIP": (6.62, 0.24044435, 0.00265, 0.07456),
    "right_middle_MCP_FE": (0.9, 0.04190723, 0.0006, 0.01276),
    "right_middle_MCP_AA": (4.76, 0.20859232, 0.00265, 0.07456),
    "right_middle_PIP": (6.62, 0.24595532, 0.00265, 0.07456),
    "right_middle_DIP": (0.9, 0.04243185, 0.0006, 0.01276),
    "right_pinky_CMC": (0.9, 0.03504461, 0.00042, 0.00378738),
    "right_pinky_MCP_FE": (4.76, 0.2085923, 0.00265, 0.07456),
    "right_pinky_MCP_AA": (6.62, 0.24595532, 0.00265, 0.07456),
    "right_pinky_PIP": (0.9, 0.04243185, 0.0006, 0.01276),
    "right_pinky_DIP": (0.9, 0.03504461, 0.00042, 0.00378738),
    "right_ring_MCP_FE": (4.76, 0.20859226, 0.00265, 0.07456),
    "right_ring_MCP_AA": (6.62, 0.24595528, 0.00265, 0.07456),
    "right_ring_PIP": (0.9, 0.04243183, 0.0006, 0.01276),
    "right_ring_DIP": (0.9, 0.0350446, 0.00042, 0.00378738),
    "right_thumb_CMC_FE": (1.38, 0.02782345, 0.00012, 0.012),
    "right_thumb_CMC_AA": (4.76, 0.20859229, 0.00265, 0.07456),
    "right_thumb_MCP_FE": (6.62, 0.24595528, 0.00265, 0.07456),
    "right_thumb_MCP_AA": (0.9, 0.04243183, 0.0006, 0.01276),
    "right_thumb_IP": (0.9, 0.0350446, 0.00042, 0.00378738),
}

HAND_JOINT_LIMITS = {
    "right_thumb_CMC_FE": (-0.1745, 1.9199),
    "right_thumb_CMC_AA": (-0.3491, 0.1309),
    "right_thumb_MCP_FE": (-0.5236, 1.3963),
    "right_thumb_MCP_AA": (-0.3491, 0.3491),
    "right_thumb_IP": (0.0, 1.7453),
    "right_index_MCP_FE": (-0.17453293, 1.5708),
    "right_index_MCP_AA": (math.radians(-15.0), math.radians(2.0)),
    "right_index_PIP": (0.0, 1.7453),
    "right_index_DIP": (0.0, 1.3963),
    "right_middle_MCP_FE": (-0.17453293, 1.5708),
    "right_middle_MCP_AA": (math.radians(-2.0), math.radians(2.0)),
    "right_middle_PIP": (0.0, 1.7453),
    "right_middle_DIP": (0.0, 1.3963),
    "right_ring_MCP_FE": (-0.17453293, 1.5708),
    "right_ring_MCP_AA": (math.radians(-2.0), math.radians(7.5)),
    "right_ring_PIP": (0.0, 1.7453),
    "right_ring_DIP": (0.0, 1.3963),
    "right_pinky_CMC": (0.0, math.radians(5.0)),
    "right_pinky_MCP_FE": (-0.17453293, 1.5708),
    "right_pinky_MCP_AA": (math.radians(-2.0), math.radians(18.0)),
    "right_pinky_PIP": (0.0, 1.7453),
    "right_pinky_DIP": (0.0, 1.3963),
}

HAND_BODY_KEYPOINTS = {
    "right_hand_C_MC": "wrist",
    "right_index_MCP_VL": "index_proximal",
    "right_index_PP": "index_proximal",
    "right_index_MP": "index_intermediate",
    "right_index_DP": "index_distal",
    "right_index_fingertip": "index_tip",
    "right_middle_MCP_VL": "middle_proximal",
    "right_middle_PP": "middle_proximal",
    "right_middle_MP": "middle_intermediate",
    "right_middle_DP": "middle_distal",
    "right_middle_fingertip": "middle_tip",
    "right_pinky_MC": "pinky_proximal",
    "right_pinky_MCP_VL": "pinky_proximal",
    "right_pinky_PP": "pinky_proximal",
    "right_pinky_MP": "pinky_intermediate",
    "right_pinky_DP": "pinky_distal",
    "right_pinky_fingertip": "pinky_tip",
    "right_ring_MCP_VL": "ring_proximal",
    "right_ring_PP": "ring_proximal",
    "right_ring_MP": "ring_intermediate",
    "right_ring_DP": "ring_distal",
    "right_ring_fingertip": "ring_tip",
    "right_thumb_CMC_VL": "thumb_proximal",
    "right_thumb_MC": "thumb_proximal",
    "right_thumb_MCP_VL": "thumb_proximal",
    "right_thumb_PP": "thumb_intermediate",
    "right_thumb_DP": "thumb_distal",
    "right_thumb_fingertip": "thumb_tip",
}
HAND_BODIES = list(HAND_BODY_KEYPOINTS)
REFERENCE_KEYPOINTS = [HAND_BODY_KEYPOINTS[name] for name in HAND_BODIES[1:]]
WRIST_BODY = HAND_BODIES[0]
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
PAD_BODIES = [f"right_{finger}_elastomer" for finger in FINGERS]
FINGERTIP_BODIES = [f"right_{finger}_fingertip" for finger in FINGERS]
NUM_HAND_DOFS = len(HAND_JOINT_DRIVE)

KEYPOINT_GROUPS = {
    "thumb_tip": [27],
    "index_tip": [5],
    "middle_tip": [10],
    "ring_tip": [21],
    "pinky_tip": [16],
    "level_1_joints": [1, 6, 11, 17, 22],
    "level_2_joints": [2, 3, 4, 7, 8, 9, 12, 13, 14, 15, 18, 19, 20, 23, 24, 25, 26],
}
