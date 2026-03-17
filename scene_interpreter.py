def detect_scene(objects):
    objects = set(objects)

    if {"bench", "tree", "person"} & objects and ("bench" in objects or "tree" in objects):
        return "outdoor public area"

    if {"refrigerator", "microwave", "oven", "sink"} & objects:
        return "kitchen-like space"

    if {"laptop", "keyboard", "mouse", "monitor"} & objects:
        return "workspace"

    if {"bed", "pillow", "blanket"} & objects:
        return "bedroom-like space"

    if {"sofa", "tv", "remote"} & objects:
        return "living room-like space"

    if {"car", "truck", "bus", "bicycle"} & objects:
        return "street or roadside area"

    if {"person"} == objects:
        return "space with a person nearby"

    if "person" in objects:
        return "shared space with people and objects"

    return "general environment"


def describe_scene(objects, mode="guide"):
    unique_objects = list(dict.fromkeys(objects))
    scene = detect_scene(unique_objects)

    if not unique_objects:
        if mode == "safety":
            return "I cannot detect any clear objects right now."
        if mode == "chill":
            return "The space around you is quiet and unclear at the moment."
        return "I cannot clearly understand the environment right now."

    object_text = ", ".join(unique_objects)

    if mode == "safety":
        return f"You seem to be in a {scene}. Detected: {object_text}."

    if mode == "chill":
        return f"This place feels like a {scene}, with {object_text} around you."

    return f"You appear to be in a {scene}. I can recognize {object_text}."