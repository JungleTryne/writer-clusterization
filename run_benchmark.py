import subprocess

configs = [
    ("./configs/encoder_crossentropy/clusterisation/c_cvl_kmeans.yaml", "./configs/encoder_crossentropy/models/font_500_unl.yaml"),
    ("./configs/encoder_crossentropy/clusterisation/c_cvl_agglr.yaml", "./configs/encoder_crossentropy/models/font_500_unl.yaml"),
    ("./configs/encoder_crossentropy/clusterisation/c_iam_kmeans.yaml", "./configs/encoder_crossentropy/models/font_500_unl.yaml"),
    ("./configs/encoder_crossentropy/clusterisation/c_iam_agglr.yaml", "./configs/encoder_crossentropy/models/font_500_unl.yaml"),
    ("./configs/encoder_crossentropy/clusterisation/c_synt_kmeans.yaml", "./configs/encoder_crossentropy/models/font_500_unl.yaml"),
    ("./configs/encoder_crossentropy/clusterisation/c_synt_agglr.yaml", "./configs/encoder_crossentropy/models/font_500_unl.yaml"),

    ("./configs/encoder_arcface/clusterisation/c_cvl_kmeans.yaml", "./configs/encoder_arcface/models/font_500_unl.yaml"),
    ("./configs/encoder_arcface/clusterisation/c_cvl_agglr.yaml", "./configs/encoder_arcface/models/font_500_unl.yaml"),
    ("./configs/encoder_arcface/clusterisation/c_iam_kmeans.yaml", "./configs/encoder_arcface/models/font_500_unl.yaml"),
    ("./configs/encoder_arcface/clusterisation/c_iam_agglr.yaml", "./configs/encoder_arcface/models/font_500_unl.yaml"),
    ("./configs/encoder_arcface/clusterisation/c_synt_kmeans.yaml", "./configs/encoder_arcface/models/font_500_unl.yaml"),
    ("./configs/encoder_arcface/clusterisation/c_synt_agglr.yaml", "./configs/encoder_arcface/models/font_500_unl.yaml"),

    # "./configs/snn/clusterisation/c_cvl_kmeans.yaml",
    # "./configs/snn/clusterisation/c_cvl_agglr.yaml",
    # "./configs/snn/clusterisation/c_iam_kmeans.yaml",
    # "./configs/snn/clusterisation/c_iam_agglr.yaml",
    # "./configs/snn/clusterisation/c_synt_kmeans.yaml",
    # "./configs/snn/clusterisation/c_synt_agglr.yaml",
]

for config in configs:
    print("===== {} & {} =====".format(*config))
    subprocess.run(["python3", "clusterize.py", "--cluster-config-path", config[0], "--model-config-path", config[1]])