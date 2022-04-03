import os

ROOT_IDDO = "/Users/iddobar-haim/Library/Mobile Documents/com~apple~CloudDocs/University/FlySightProject/RealInputClips2"
#ROOT ="/Users/elior/Library/Mobile Documents/com~apple~CloudDocs/FlySightProject/RealInputClips"
ROOT_ELIOR = "/Users/elior/Library/Mobile Documents/com~apple~CloudDocs/FlySightProject/RealInputClips2"
PILLAR = "Pillar(A)"
CORNER = "Corner(B)"
EDGE = "WallEdge(C)"
OBJECT_DIRS = (PILLAR, CORNER, EDGE)


def all_clips(function, input_root_directory=ROOT_IDDO, *args, **kwargs):
    """
    Applies a given function to all .mp4 clips found in a given directory.
    :param function: The function to apply to all clips found
    :param input_root_directory: The directory to recursively search for clips in.
    :param args: Any positional arguments to pass to the given function.
    :param kwargs: Any keyword arguments to pass to the given function.
    """
    for odir in OBJECT_DIRS:
        for root, dirs, files in os.walk(os.path.join(input_root_directory, odir)):
            for name in dirs:
                _make_directory_for_output(function, name, root)

            for name in files:
                if not len(dirs):
                    break
                if os.path.splitext(name)[1] == ".mp4":
                    _run_function_on_clip(function, root, name, args, kwargs)


def _run_function_on_clip(function, root_directory, clip_name, args, kwargs):
    current_clip = os.path.join(root_directory, clip_name)
    print(current_clip)
    out_dir = os.path.join(os.path.split(current_clip)[0], function.__name__ + '_func')
    function(output_dir=out_dir, input_clip=current_clip, *args, **kwargs)


def _make_directory_for_output(function, name, root):
    print(os.path.join(root, name))
    if not os.path.exists(os.path.join(root, name, function.__name__ + "_func")):
        print("not there")
        if not name.endswith("_func"):
            os.makedirs(os.path.join(root, name, function.__name__ + "_func"))
    else:
        print("there")


def make_txt_file_with_mp4_name(output_dir, input_clip):
    """For Testing."""
    txt_file_name = os.path.splitext(os.path.split(input_clip)[1])[0]
    with open(os.path.join(output_dir, txt_file_name + ".txt"), 'w') as f:
        f.write("HI!!!")
        print(f.name)


if __name__ == '__main__':
    all_clips(make_txt_file_with_mp4_name, ROOT_IDDO)
