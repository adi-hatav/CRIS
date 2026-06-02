import argparse
import os

from options import DataOptions
import inference
import training


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser(description="CRIS training and isotropic reconstruction")
    opt = DataOptions().parse(parser=main_parser)
    DataOptions().print_options(opt)

    model_path = os.path.join(opt.main_dir, "models", "best_model.pth")
    saving_dir = opt.main_dir

    if opt.phase == "train":
        training.train(opt)

    if opt.phase == "evaluation":
        inference.export_isotropic_volumes(
            opt,
            opt.default_plane,
            model_path,
            saving_dir,
        )

    if opt.export_isotropic_volumes and opt.phase == "train":
        opt.phase = "evaluation"
        opt.pre_trained = opt.main_dir
        inference.export_isotropic_volumes(
            opt,
            opt.default_plane,
            model_path,
            saving_dir,
        )
