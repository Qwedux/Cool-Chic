import os

from util.util import Results_Params, dt_to_intkey, list_dict_to_dict_list


class AnalysisHandler:
    def __init__(self, data_dir: str) -> None:
        """Initialize the analysis handler.

        Args:
            data_dir (str): The directory containing the data.
                Should contain folders results, train_logs and trained_models.
        """
        self.data_dir = data_dir
        self.results_dir = os.path.join(data_dir, "results")
        self.train_logs_dir = os.path.join(data_dir, "train_logs")
        self.trained_models_dir = os.path.join(data_dir, "trained_models")

    def load_results(self) -> list[str]:
        """Load the results from the data directory.

        Returns:
            dict: A dictionary containing the results.
        """
        res_files_unfiltered = [f for f in os.listdir(self.results_dir) if f.endswith(".log")]
        res_files_unfiltered.sort()
        dates, times = zip(*(s.split("__")[:2] for s in res_files_unfiltered))
        creation_times = {
            index: dt_to_intkey(date, time) for index, (date, time) in enumerate(zip(dates, times))
        }
        only_after = dt_to_intkey("2025_10_30", "00_00_00")
        res_files = [
            res_files_unfiltered[index]
            for index, creation_time in creation_times.items()
            if creation_time >= only_after
        ]
        print(f"Loaded {len(res_files)} results files.")
        return res_files

    def parse_results(
        self, res_files: list[str]
    ) -> tuple[list[dict[str, float | str]], list[float], Results_Params]:
        """Parse the results from the result files into a dictionary of results.

        Args:
            res_files (list[str]): The list of result files to process.

        Returns:
            tuple[list[dict[str, float | str]], list[float], Resutls_Params]: A tuple containing the raw results, the rate of the image bistreams and the results parameters.
        """
        results_params = Results_Params()
        rate_img_bistream: list[float] = []

        with open(os.path.join(self.results_dir, res_files[0]), "r") as infile:
            lines = infile.readlines()
            for line in lines:
                if line.startswith("Using color space"):
                    results_params.using_color_space = (
                        line[len("Using color space") :].strip().split(" with bitdepths")[0]
                    )
                if line.startswith("Using image ARM:"):
                    results_params.using_image_arm = (
                        line[len("Using image ARM:") :].strip() == "True"
                    )
                if line.startswith("Using multi-region image ARM:"):
                    results_params.using_multi_region_image_arm_setup = (
                        line[len("Using multi-region image ARM:") :].strip().split(" ")[-1]
                    )
                if line.startswith("Using encoder gain:"):
                    results_params.using_encoder_gain = int(
                        line[len("Using encoder gain:") :].strip()
                    )
                if line.startswith("Using multi-region image ARM:"):
                    results_params.using_multi_region_image_arm = (
                        line[len("Using multi-region image ARM:") :].strip() == "True"
                    )
                if line.startswith("Using color regression:"):
                    results_params.using_color_regression = (
                        line[len("Using color regression:") :].strip() == "True"
                    )
                if line.startswith("Total training iterations:"):
                    results_params.total_training_iterations = int(
                        line[len("Total training iterations:") :].strip()
                    )
                if line.startswith("Total MAC per pixel:"):
                    results_params.total_mac_per_pixel = float(
                        line[len("Total MAC per pixel:") :].strip()
                    )
                if line.startswith("Image index"):
                    results_params.image_index = int(line[len("Image index:") :].strip())

            if results_params.using_color_space == "":
                print(f"Could not determine color space for file {res_files[0]}, skipping.")
                raise ValueError("No color space found.")

        raw_results: list[dict[str, float | str]] = []
        for res_file in res_files:
            with open(os.path.join(self.results_dir, res_file), "r") as infile:
                lines = infile.readlines()
                for line in lines:
                    if line.startswith("Rate Img bistream:"):
                        rate_img_bistream.append(float(line[len("Rate Img bistream:") :].strip()))
                    if "Final results after quantization:" in line:
                        parts = line[len("Final results after quantization:") :].strip().split(", ")
                        raw_results.append({})
                        raw_results[-1]["Im_name"] = res_file.split("_")[-3][:10]
                        raw_results[-1]["taskId"] = f"{res_file.split('_')[-1].split('.')[0]:>6}"
                        for part in parts:
                            key, value = part.split(": ")
                            raw_results[-1][key] = float(value)

        return raw_results, rate_img_bistream, results_params

    def postprocess_results(self, raw_results: list[dict[str, float | str]]) -> dict[str, list[float | str]]:
        """Clean the raw results by fixing what is wrong with the individual runs.

        As many cluster runs resulted in unclean data, e.g. duplicated runs, bad rate_nn values, etc, we need to fix them here.
        Some fixes may be needed during the parsing part, if so they are denoted as such.
        """
        cleaned_results = list_dict_to_dict_list(raw_results)
        def sort_helper(table: dict[str, list[float | str]]) -> dict[str, list[float | str]]:
            n_rows = len(next(iter(table.values())))
            losses = table["Loss"]
            indices = list(range(n_rows))
            indices.sort(key=lambda i: losses[i])
            sorted_table = {key: [table[key][i] for i in indices] for key in table.keys()}
            return sorted_table
        cleaned_results = sort_helper(cleaned_results)
        return cleaned_results
        