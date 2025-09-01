import os
from pathlib import Path
from omegaconf import DictConfig
import hydra
from torchvision import transforms

from src.data.dataloader import create_train_val_dataloader
from src.data.dataset import Space_dataset

from src.utils.seeding import set_seed

# Определяем корень и путь к configs
project_root = Path(__file__).parent.parent
config_path = str(project_root / "configs")

os.chdir(project_root)

transform = transforms.Compose([
    transforms.ToTensor(),               # обязательно!
    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # опционально
])

@hydra.main(config_path=config_path, config_name="base", version_base=None)
def main(cfg: DictConfig):
    path_csv = cfg.data.csv_path
    img_path = cfg.data.img_path
    seed = cfg.seed
    set_seed(seed)
    reg = create_train_val_dataloader(path_csv,img_path,transform)
    for c in reg:
        print(c)
    # dataset = Space_dataset(path_csv=path_csv,path_img=img_path,list_label=['cod_class','cod_subclass'],list_x=['ra','dec'])
    # print("CSV Path:", dataset)

if __name__ == "__main__":
    main()