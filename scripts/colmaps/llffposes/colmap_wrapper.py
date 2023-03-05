import os
import subprocess


# $ DATASET_PATH=/path/to/dataset

# $ colmap feature_extractor \
#    --database_path $DATASET_PATH/database.db \
#    --image_path $DATASET_PATH/images

# $ colmap exhaustive_matcher \
#    --database_path $DATASET_PATH/database.db

# $ mkdir $DATASET_PATH/sparse

# $ colmap mapper \
#     --database_path $DATASET_PATH/database.db \
#     --image_path $DATASET_PATH/images \
#     --output_path $DATASET_PATH/sparse

# $ mkdir $DATASET_PATH/dense
colmap_path = "D:\\MSI_NB\\source\\util\\COLMAP-3.6-exe\\COLMAP.bat"


def run_colmap(basedir, match_type, pipeline, imagedir='images', share_intrin=True):
    logfile_name = os.path.join(basedir, 'colmap_output.txt')
    logfile = open(logfile_name, 'w')

    if "feature_extractor" in pipeline:
        feature_extractor_args = [
            colmap_path, 'feature_extractor',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, imagedir),
            '--ImageReader.camera_model', 'SIMPLE_PINHOLE'
            # '--SiftExtraction.use_gpu', '0',
        ]
        if share_intrin:
            feature_extractor_args += ['--ImageReader.single_camera', '1']
        feat_output = (subprocess.check_output(feature_extractor_args, universal_newlines=True))
        logfile.write(feat_output)
        print('Features extracted')

    if "matcher" in pipeline:
        exhaustive_matcher_args = [
            colmap_path, match_type,
            '--database_path', os.path.join(basedir, 'database.db'),
        ]

        match_output = (subprocess.check_output(exhaustive_matcher_args, universal_newlines=True))
        logfile.write(match_output)
        print('Features matched')

    if "mapper" in pipeline:
        p = os.path.join(basedir, 'sparse')
        if not os.path.exists(p):
            os.makedirs(p)

        # mapper_args = [
        #     'colmap', 'mapper',
        #         '--database_path', os.path.join(basedir, 'database.db'),
        #         '--image_path', os.path.join(basedir, 'images'),
        #         '--output_path', os.path.join(basedir, 'sparse'),
        #         '--Mapper.num_threads', '16',
        #         '--Mapper.init_min_tri_angle', '4',
        # ]
        mapper_args = [
            colmap_path, 'mapper',
            '--database_path', os.path.join(basedir, 'database.db'),
            '--image_path', os.path.join(basedir, imagedir),
            '--output_path', os.path.join(basedir, 'sparse'),  # --export_path changed to --output_path in colmap 3.6
            '--Mapper.num_threads', '12',
            '--Mapper.init_min_tri_angle', '4',
            '--Mapper.multiple_models', '0',
            # '--Mapper.extract_colors', '0',
        ]

        map_output = (subprocess.check_output(mapper_args, universal_newlines=True))

        logfile.write(map_output)

    if "convert" in pipeline:
        converter_args = [
            colmap_path, 'model_converter',
            '--input_path', os.path.join(basedir, 'sparse/0'),
            '--output_path', os.path.join(basedir, 'sparse/0'),
            '--output_type', 'TXT',
        ]

        converter_output = (subprocess.check_output(converter_args, universal_newlines=True))
        print('Txt model converted')

        logfile.write(converter_output)
    logfile.close()
    print('Sparse map created')

    print('Finished running COLMAP, see {} for logs'.format(logfile_name))
