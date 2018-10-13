import os
import re
import pandas as pd
from functools import partial
from Data.utils_prepare_data import *
from sklearn.preprocessing import scale


def load_patient(subj, tmpl):
    if tmpl.endswith('.nii'):
        data_path = format_config(tmpl, {
            'subject': subj,
        })
        if not os.path.exists(data_path):
            data_path = format_config(tmpl, {
                'subject': subj.split('_')[-1],
            })
        data = load_nifti_data(data_path, mask=True, normalization=True)
    elif tmpl.endswith('.1D'):
        df = pd.read_csv(format_config(tmpl, {
            'subject': subj,
        }), sep="\t", header=0)
        df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

        ROIs = ["#" + str(y) for y in sorted([int(x[1:]) for x in df.keys().tolist()])]

        functional = np.nan_to_num(df[ROIs].as_matrix().T).tolist()
        functional = scale(functional, axis=1)
        functional = compute_connectivity(functional)
        functional = functional.astype(np.float32)
        data = functional

    return subj, data


def load_patients(subjs, tmpl, jobs=10):
    partial_load_patient = partial(load_patient, tmpl=tmpl)
    msg = 'Processing {current} of {total}'
    return dict(run_progress(partial_load_patient, subjs, message=msg, jobs=jobs))


def load_patients_to_file(hdf5, features, datasets):
    for dataset in datasets:
        pheno = load_phenotypes(dataset)
        download_root = 'F:/OneDriveOffL/Data/Data/{:s}/Results'.format(dataset)
        features_path = {
            'falff': 'fALFF_FunImgARCW/fALFFMap_{subject}.nii',
            'vmhc': 'VMHC_FunImgARCWFsymS/zVMHCMap_{subject}.nii',
            'reho': 'ReHo_FunImgARCWF/ReHoMap_{subject}.nii',
        }
        dataset_group = hdf5.require_group('{:s}/subjects'.format(dataset))
        file_ids = pheno['FILE_ID']

        for feature in features:
            file_template = os.path.join(download_root, features_path[feature])
            data = load_patients(file_ids, tmpl=file_template)

            for pid in data:
                record = pheno[pheno['FILE_ID'] == pid].iloc[0]
                subject_storage = dataset_group.require_group(pid)
                subject_storage.attrs['id'] = record['FILE_ID']
                subject_storage.attrs['y'] = record['DX_GROUP']
                subject_storage.attrs['site'] = record['SITE']
                subject_storage.attrs['sex'] = record['SEX']
                create_dataset_hdf5(group=subject_storage,
                                    name=feature,
                                    data=data[pid],
                                    )


def load_phenotypes(dataset: str) -> pd.DataFrame:
    if dataset == 'ABIDE':
        pheno = load_phenotypes_ABIDE_RfMRIMaps()
    elif dataset == 'ADHD':
        pheno = load_phenotypes_ADHD_RfMRIMaps()
    elif dataset == 'ABIDE II':
        pheno = load_phenotypes_ABIDE2_RfMRIMaps()
    elif dataset == 'FCP':
        pheno = load_phenotypes_FCP_RfMRIMaps()
    return pheno


def load_phenotypes_ABIDE_RfMRIMaps():
    pheno_path = 'F:/OneDriveOffL/Data/Data/ABIDE/RfMRIMaps_ABIDE_Phenotypic.csv'
    pheno = pd.read_csv(pheno_path)

    pheno['FILE_ID'] = pheno['SUB_ID'].apply(lambda v: '00{:}'.format(v))
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v) - 1)
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['SITE'] = pheno['SITE_ID']
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT',
                  ]]


def load_phenotypes_ABIDE2_RfMRIMaps():
    pheno_path = 'F:/OneDriveOffL/Data/Data/ABIDE II/RfMRIMaps_ABIDE2_Phenotypic.csv'
    pheno = pd.read_csv(pheno_path)

    pheno['FILE_ID'] = pheno['SUB_LIST']
    pheno['SITE'] = pheno['SUB_LIST'].apply(lambda v: '_'.join(v.split('-')[1].split('_')[0:2]))
    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v) - 1)
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT'
                  ]]


def load_phenotypes_abide_cyberduck():
    pheno_path = 'F:/OneDriveOffL/Data/Data/ABIDE/RfMRIMaps_ABIDE_Phenotypic.csv'
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']

    pheno['DX_GROUP'] = pheno['DX_GROUP'].apply(lambda v: int(v) - 1)
    pheno['SITE'] = pheno['SITE_ID'].apply(lambda v: re.sub('_[0-9]', '', v))
    pheno['SEX'] = pheno['SEX'].apply(lambda v: {1: "M", 2: "F"}[v])
    pheno['MEAN_FD'] = pheno['func_mean_fd']
    pheno['SUB_IN_SMP'] = pheno['SUB_IN_SMP'].apply(lambda v: v == 1)
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'MEAN_FD',
                  'SUB_IN_SMP',
                  'STRAT']]


def load_phenotypes_ADHD_RfMRIMaps():
    pheno_path = 'F:\OneDriveOffL\Data\Data\ADHD\Phenotypic_ADHD.csv'
    pheno = pd.read_csv(pheno_path)
    pheno = pheno[pheno['DX'] != 'pending']

    pheno['FILE_ID'] = pheno['Participant ID']
    pheno['DX_GROUP'] = pheno['DX'].apply(lambda v: 1 if int(v) > 0 else 0)
    pheno['SITE'] = pheno['Participant ID'].apply(lambda v: '_'.join(v.split('_')[1:-1]))
    pheno['SEX'] = pheno['Gender'].apply(lambda v: {1: 'F', 0: 'M', 2: 'U', }[v])
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT',
                  ]]


def load_phenotypes_FCP_RfMRIMaps():
    pheno_path = 'F:\OneDriveOffL\Data\Data\FCP\FCP_RfMRIMaps_Info.csv'
    pheno = pd.read_csv(pheno_path)

    pheno['FILE_ID'] = pheno['Subject ID']
    pheno['DX_GROUP'] = pheno['DX']
    pheno['SITE'] = pheno['Site']
    pheno['SEX'] = pheno['Sex']
    pheno['STRAT'] = pheno[['SITE', 'DX_GROUP']].apply(lambda x: "_".join([str(s) for s in x]), axis=1)

    pheno.index = pheno['FILE_ID']

    return pheno[['FILE_ID',
                  'DX_GROUP',
                  'SEX',
                  'SITE',
                  'STRAT'
                  ]]
