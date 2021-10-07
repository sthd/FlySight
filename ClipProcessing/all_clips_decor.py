import os

ROOT = "/Users/iddobar-haim/Library/Mobile Documents/com~apple~CloudDocs/FlySightProject/RealInputClips"
PILLAR = "Pillar(A)"
CORNER = "Corner(B)"
EDGE = "WallEdge(C)"
OBJECT_DIRS = (PILLAR, CORNER, EDGE)


def all_clips(function, *args, **kwargs):
    for odir in OBJECT_DIRS:
        for root, dirs, files in os.walk(os.path.join(ROOT, odir)):
            for name in dirs:
                print(os.path.join(root, name))
                if not os.path.exists(os.path.join(root, name, function.__name__ + "_func")):
                    print("not there")
                    if not name.endswith("_func"):
                        os.makedirs(os.path.join(root, name, function.__name__ + "_func"))
                else:
                    print("there")

            for name in files:
                if not len(dirs):
                    break
                if os.path.splitext(name)[1] == ".mp4":
                    current_clip = os.path.join(root, name)
                    print(current_clip)
                    out_dir = os.path.join(os.path.split(current_clip)[0], function.__name__ + '_func')
                    function(output_dir=out_dir, input_clip=current_clip, *args, **kwargs)


def make_txt_file_with_mp4_name(output_dir, input_clip):
    txt_file_name = os.path.splitext(os.path.split(input_clip)[1])[0]
    with open(os.path.join(output_dir, txt_file_name + ".txt"), 'w') as f:
        f.write("HI!!!")
        print(f.name)


if __name__ == '__main__':
    all_clips(make_txt_file_with_mp4_name)
