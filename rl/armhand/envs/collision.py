MOUNT_EXCLUSIONS = (("Lconnector", "hand_wrist"),)
ADJACENT_EXCLUSIONS = (
    ("hand_C_MC", "thumb_CMC_VL"),
    ("hand_C_MC", "thumb_MC"),
    ("hand_C_MC", "index_MCP_VL"),
    ("hand_C_MC", "index_PP"),
    ("hand_C_MC", "middle_MCP_VL"),
    ("hand_C_MC", "middle_PP"),
    ("hand_C_MC", "ring_MCP_VL"),
    ("hand_C_MC", "ring_PP"),
    ("hand_C_MC", "pinky_MC"),
    ("index_MCP_VL", "index_PP"),
    ("index_PP", "index_MP"),
    ("index_MP", "index_DP"),
    ("middle_MCP_VL", "middle_PP"),
    ("middle_PP", "middle_MP"),
    ("middle_MP", "middle_DP"),
    ("ring_MCP_VL", "ring_PP"),
    ("ring_PP", "ring_MP"),
    ("ring_MP", "ring_DP"),
    ("pinky_MC", "pinky_MCP_VL"),
    ("pinky_MC", "pinky_PP"),
    ("pinky_MCP_VL", "pinky_PP"),
    ("pinky_PP", "pinky_MP"),
    ("pinky_MP", "pinky_DP"),
    ("thumb_CMC_VL", "thumb_MC"),
    ("thumb_MC", "thumb_MCP_VL"),
    ("thumb_MC", "thumb_PP"),
    ("thumb_MCP_VL", "thumb_PP"),
    ("thumb_PP", "thumb_DP"),
)


def neighbor_filter_masks(body_shapes, excluded_pairs, body_order, first_bit):
    nodes = {body for pair in excluded_pairs for body in pair}
    ordered = [body for body in dict.fromkeys(body_order) if body in nodes]
    bits = {body: 1 << first_bit + offset for offset, body in enumerate(ordered)}
    body_masks = dict(bits)
    for body_a, body_b in excluded_pairs:
        body_masks[body_a] |= bits[body_b]
        body_masks[body_b] |= bits[body_a]
    shape_masks = {}
    for body, mask in body_masks.items():
        for shape_id in body_shapes[body]:
            shape_masks[shape_id] = mask
    return shape_masks, len(ordered)


def pair_filter_masks(body_shapes, excluded_pairs, first_bit):
    shape_masks = {}
    for offset, pair in enumerate(excluded_pairs):
        bit = 1 << first_bit + offset
        for body in pair:
            for shape_id in body_shapes[body]:
                shape_masks[shape_id] = shape_masks.get(shape_id, 0) | bit
    return shape_masks


def hand_self_collision_masks(body_names, body_shapes):
    shape_masks, used_bits = neighbor_filter_masks(
        body_shapes,
        [("right_" + a, "right_" + b) for a, b in ADJACENT_EXCLUSIONS],
        body_names,
        first_bit=2,
    )
    next_bit = 2 + used_bits
    mount_masks = pair_filter_masks(body_shapes, [(a, "right_" + b) for a, b in MOUNT_EXCLUSIONS], first_bit=next_bit)
    for shape_id, mask in mount_masks.items():
        shape_masks[shape_id] = shape_masks.get(shape_id, 0) | mask
    next_bit += len(MOUNT_EXCLUSIONS)
    arm_bodies = [
        name
        for name in body_names
        if (name == "base" or name.startswith("fr3") or name == "Lconnector") and body_shapes[name]
    ]
    group_bit = 1 << next_bit
    for body in dict.fromkeys(arm_bodies):
        for shape_id in body_shapes[body]:
            shape_masks[shape_id] = shape_masks.get(shape_id, 0) | group_bit
    return shape_masks
