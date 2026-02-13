import datetime


def dt_to_intkey(date_str: str, time_str: str) -> float:
    dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y_%m_%d %H_%M_%S")
    return dt.timestamp()


def list_dict_to_dict_list(
    lst: list[dict[str, float | str]],
) -> dict[str, list[float | str]]:
    if not lst:
        return {}
    keys = lst[0].keys()
    dict_list = {key: [] for key in keys}
    for d in lst:
        for key in keys:
            dict_list[key].append(d.get(key, 0.0))
    return dict_list

class Results_Params:
    def __init__(self) -> None:
        self.image_index: int | None = None
        self.using_color_space: str | None = None
        self.using_image_arm: bool | None = None
        self.using_encoder_gain: int | None = None
        self.using_multi_region_image_arm: bool | None = None
        self.using_multi_region_image_arm_setup: str | None = None
        self.using_color_regression: bool | None = None
        self.total_training_iterations: int | None = None
        self.total_mac_per_pixel: float | None = None
    
    def __str__(self) -> str:
        return "\n".join([f"{key}: {value}" for key, value in self.__dict__.items()])
    
    def __repr__(self) -> str:
        return self.__str__()
