window.GALATEA_PIPELINE_DATA = {
  "samples": [
    {
      "id": "mouse",
      "label": "Mouse",
      "sourceTag": "long2_0904_mouse_01_seed904160174",
      "frames": 145,
      "fps": 24,
      "contact": [
        [
          34,
          125
        ]
      ],
      "reviewedBoundaries": false,
      "execution": {
        "src": "./videos/highlights/mouse-real-world-sync.mp4",
        "poster": "./videos/highlights/mouse-poster.jpg",
        "duration": 6.041667,
        "action": "Push and slide"
      },
      "syncKeyframes": [[0, 0], [1.2, 1.2], [2.0, 2.0], [3.0, 3.0], [4.5, 4.5], [6.041667, 6.041667]],
      "media": {
        "pipeline": {
          "src": "./videos/pipeline-showcase/mouse/pipeline-hq.mp4",
          "poster": "./videos/pipeline-showcase/mouse/pipeline-hq.jpg"
        },
        "input": {
          "src": "./videos/pipeline-showcase/mouse/input.mp4",
          "poster": "./videos/pipeline-showcase/mouse/input.jpg"
        },
        "depth": {
          "src": "./videos/pipeline-showcase/mouse/depth.mp4",
          "poster": "./videos/pipeline-showcase/mouse/depth.jpg"
        },
        "sam2": {
          "src": "./videos/pipeline-showcase/mouse/sam2.mp4",
          "poster": "./videos/pipeline-showcase/mouse/sam2.jpg"
        },
        "trajectory": {
          "src": "./videos/pipeline-showcase/mouse/trajectory.mp4",
          "poster": "./videos/pipeline-showcase/mouse/trajectory.jpg"
        }
      },
      "notes": ""
    },
    {
      "id": "drill",
      "label": "Power Drill",
      "sourceTag": "long0904_drill_19_remaining_v3_seed904273002",
      "frames": 169,
      "fps": 24,
      "contact": [
        [
          38,
          168
        ]
      ],
      "reviewedBoundaries": true,
      "execution": {
        "src": "./videos/highlights/drill-real-world-sync.mp4",
        "poster": "./videos/highlights/drill-poster.jpg",
        "duration": 7.041667,
        "action": "Grasp and pose adjustment"
      },
      "syncKeyframes": [
        [0, 0], [1.25, 1.25], [2.25, 2.25],
        [2.583333, 2.583333], // Lift begins.
        [3.041667, 3.041667], [3.666667, 3.666667],
        [4.291667, 4.291667], // Apex.
        [4.458333, 4.458333], // Downward motion begins.
        [5.166667, 5.166667], [5.833333, 5.833333], [7.041667, 7.041667]
      ],
      "media": {
        "pipeline": {
          "src": "./videos/pipeline-showcase/drill/pipeline-hq.mp4",
          "poster": "./videos/pipeline-showcase/drill/pipeline-hq.jpg"
        },
        "input": {
          "src": "./videos/pipeline-showcase/drill/input.mp4",
          "poster": "./videos/pipeline-showcase/drill/input.jpg"
        },
        "depth": {
          "src": "./videos/pipeline-showcase/drill/depth.mp4",
          "poster": "./videos/pipeline-showcase/drill/depth.jpg"
        },
        "sam2": {
          "src": "./videos/pipeline-showcase/drill/sam2.mp4",
          "poster": "./videos/pipeline-showcase/drill/sam2.jpg"
        },
        "trajectory": {
          "src": "./videos/pipeline-showcase/drill/trajectory.mp4",
          "poster": "./videos/pipeline-showcase/drill/trajectory.jpg"
        }
      },
      "notes": ""
    },
    {
      "id": "hammer",
      "label": "God Hammer",
      "sourceTag": "long0904_godhammer_12_remaining_v3_seed904271896",
      "frames": 169,
      "fps": 24,
      "contact": [
        [
          38,
          168
        ]
      ],
      "reviewedBoundaries": true,
      "execution": {
        "src": "./videos/highlights/hammer-real-world-sync.mp4",
        "poster": "./videos/highlights/hammer-poster.jpg",
        "duration": 7.041667,
        "action": "Grasp and lift"
      },
      "syncKeyframes": [
        [0, 0], [1.25, 1.25], [2.25, 2.25],
        [3.083333, 3.083333], // Lift begins.
        [3.75, 3.75], [4.25, 4.25],
        [4.583333, 4.583333], // Apex.
        [4.75, 4.75], // Downward motion begins.
        [5.166667, 5.166667], [5.875, 5.875], [7.041667, 7.041667]
      ],
      "media": {
        "pipeline": {
          "src": "./videos/pipeline-showcase/hammer/pipeline-hq.mp4",
          "poster": "./videos/pipeline-showcase/hammer/pipeline-hq.jpg"
        },
        "input": {
          "src": "./videos/pipeline-showcase/hammer/input.mp4",
          "poster": "./videos/pipeline-showcase/hammer/input.jpg"
        },
        "depth": {
          "src": "./videos/pipeline-showcase/hammer/depth.mp4",
          "poster": "./videos/pipeline-showcase/hammer/depth.jpg"
        },
        "sam2": {
          "src": "./videos/pipeline-showcase/hammer/sam2.mp4",
          "poster": "./videos/pipeline-showcase/hammer/sam2.jpg"
        },
        "trajectory": {
          "src": "./videos/pipeline-showcase/hammer/trajectory.mp4",
          "poster": "./videos/pipeline-showcase/hammer/trajectory.jpg"
        }
      },
      "notes": "User-selected local grasp variant B; residual hand–object interpenetration remains."
    }
  ],
  "contactAttribution": "HOI-DETR initialization followed by pipeline temporal calibration",
  "mediaVersion": "stream-optimized-20260909",
  "frameLayout": {
    "width": 1272,
    "height": 720,
    "regions": {
      "input": [
        0,
        0,
        848,
        480
      ],
      "depth": [
        848,
        0,
        424,
        240
      ],
      "sam2": [
        848,
        240,
        424,
        240
      ],
      "trajectory": [
        0,
        480,
        424,
        240
      ]
    }
  }
};
