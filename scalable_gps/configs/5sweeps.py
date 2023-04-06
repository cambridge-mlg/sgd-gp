from configs.default import get_config as get_default_config


def get_config(config_string):
    config = get_default_config(config_string)
    
    if config.dataset_name == "pol":
        config.kernel_config.length_scale = [
            0.51726147, 0.61132019, 1.45746202, 9.33419981, 1.71735754, 4.99153748, 5.16781788, 9.14996109, 9.85517597,
            6.35742264, 3.48656049, 2.87244892, 4.99046278, 8.24322233, 7.44450874, 3.71949692, 8.49756145, 9.69921303,
            9.44414692, 9.40644875, 8.76341991, 10.01268349, 10.23144112, 5.75442648, 7.26389265, 4.80250015]
        
        config.kernel_config.signal_scale = 0.3939051330089569
        
        config.kernel_config.noise_scale = 0.03657480329275131
    if config.dataset_name == "protein":
        config.kernel_config.length_scale = [0.39719116, 2.70550394, 0.8316101, 0.21243448, 0.37867384, 0.315153,
                                             0.12718033, 0.35584376, 0.27604335]
        config.kernel_config.signal_scale = 0.8629741787910461
        config.kernel_config.noise_scale = 0.07735464163124561
        
    return config
    