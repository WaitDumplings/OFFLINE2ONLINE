import yaml

class Config:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        self.data = data

        # intial cus_num, rs_num
        # self.cus_num = self.data.get('num_customers', None)
        # self.rs_num = self.data.get('num_charging_stations', None)


    def yaml_to_dict(self, data: dict) -> dict:
        config_dict = {}
        for key, value in data.items():
            if isinstance(value, dict):
                config_dict[key] = self.yaml_to_dict(value)
            else:
                config_dict[key] = value
        return config_dict

    def setup_env_parameters(self) -> dict:
        """Reccurent YAML -> Python Dict"""
        return self.yaml_to_dict(self.data)


def main():
    # Load Config
    config_path = './config.yaml'
    config = Config(config_path)
    config_dict = config.load_config()
    
    print("Loaded YAML config:\n")
    print(config_dict)

if __name__ == "__main__":
    main()
