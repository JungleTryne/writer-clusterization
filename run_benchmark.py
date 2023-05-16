import subprocess

configs = [
    "./configs/encoder_arcface/clusterisation/c_cvl_kmeans.yaml",
    "./configs/encoder_arcface/clusterisation/c_cvl_agglr.yaml",
    "./configs/encoder_arcface/clusterisation/c_iam_kmeans.yaml",
    "./configs/encoder_arcface/clusterisation/c_iam_agglr.yaml",
    "./configs/encoder_arcface/clusterisation/c_synt_kmeans.yaml",
    "./configs/encoder_arcface/clusterisation/c_synt_agglr.yaml",

    "./configs/encoder_arcface/clusterisation_eucl/c_cvl_kmeans.yaml",
    "./configs/encoder_arcface/clusterisation_eucl/c_cvl_agglr.yaml",
    "./configs/encoder_arcface/clusterisation_eucl/c_iam_kmeans.yaml",
    "./configs/encoder_arcface/clusterisation_eucl/c_iam_agglr.yaml",
    "./configs/encoder_arcface/clusterisation_eucl/c_synt_kmeans.yaml",
    "./configs/encoder_arcface/clusterisation_eucl/c_synt_agglr.yaml",

    # "./configs/encoder_arcface/clusterisation/c_synt_meanshift.yaml",
    # "./configs/encoder_arcface/clusterisation/c_cvl_meanshift.yaml",
    # "./configs/encoder_arcface/clusterisation/c_iam_meanshift.yaml",

    "./configs/encoder_crossentropy/clusterisation/c_cvl_kmeans.yaml",
    "./configs/encoder_crossentropy/clusterisation/c_cvl_agglr.yaml",
    "./configs/encoder_crossentropy/clusterisation/c_iam_kmeans.yaml",
    "./configs/encoder_crossentropy/clusterisation/c_iam_agglr.yaml",
    "./configs/encoder_crossentropy/clusterisation/c_synt_kmeans.yaml",
    "./configs/encoder_crossentropy/clusterisation/c_synt_agglr.yaml",

    # "./configs/encoder_crossentropy/clusterisation/c_synt_meanshift.yaml",
    # "./configs/encoder_crossentropy/clusterisation/c_cvl_meanshift.yaml",
    # "./configs/encoder_crossentropy/clusterisation/c_iam_meanshift.yaml",
]

for config in configs:
    print("===== {} =====".format(config))
    subprocess.run(["python3", "clusterize.py", "--cluster-config-path", config])