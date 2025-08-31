import os
from pathlib import Path
from omegaconf import DictConfig
import hydra

from src.data.dataset import Space_dataset

from src.utils.seeding import set_seed

# Определяем корень и путь к configs
project_root = Path(__file__).parent.parent
config_path = str(project_root / "configs")

os.chdir(project_root)


@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    path_img = cfg.data.csv_path
    img_path = cfg.data.img_path
    seed = cfg.seed
    set_seed(seed)

    dataset = Space_dataset(path_csv=img_path,path_img=path_img,list_label=['cod_class','cod_subclass'],list_x=['ra','dec'])
    print("CSV Path:", dataset)

if __name__ == "__main__":
    main()