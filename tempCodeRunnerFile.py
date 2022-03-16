print("Loading data...")
data = load_data(1000)
encoder = VarEncoder([376,500,1], 
        conv_filters=(8, 8, 4),
        conv_kernels=(3, 3, 3),
        conv_strides=(2, 2, 2),
        latent_space_dim=69)
encoder.compile()
encoder.train(data, 32, 20)
encoder.save("dogs1k_20")
