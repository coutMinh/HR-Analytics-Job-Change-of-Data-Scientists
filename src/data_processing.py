"""
HR Analytics Data Preprocessing (NumPy-only)
Strategy: DROP (id, city) | SCALE (2) | ENCODE (5) | ONE-HOT (4) | ENGINEER (6)
"""
import numpy as np
import json
from typing import Dict, List, Tuple, Optional

# ===== MAPPINGS =====
EXP_BINS = {**dict.fromkeys(['<1','1','2'], 0), **dict.fromkeys(['3','4','5'], 1),
            **dict.fromkeys(['6','7','8','9','10'], 2), **dict.fromkeys(['11','12','13','14','15'], 3),
            **dict.fromkeys(['16','17','18','19','20','>20'], 4)}
COMP_SIZE = {'<10':0, '10/49':1, '50-99':2, '100-500':3, '500-999':4, '1000-4999':5, '5000-9999':6, '10000+':7}
LAST_JOB = {'never':0, '1':1, '2':2, '3':3, '4':4, '>4':5}

# ===== UTILITIES =====
def load_csv(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=str, encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as f:
        headers = f.readline().strip().split(',')
    return data, headers

def get_col(data, headers, name):
    return data[:, headers.index(name)]

def to_numeric(arr, missing=-999.0):
    result = np.full(len(arr), missing, dtype=np.float64)
    for i, v in enumerate(arr):
        if v.strip():
            try: result[i] = float(v)
            except: pass
    return result

# ===== ENCODERS =====
def encode_gender(col):
    """Male=0, Female=1, Other/Missing=2"""
    return np.array([0 if v=='Male' else 1 if v=='Female' else 2 for v in col], dtype=np.int32)

def encode_binary(col):
    """Has relevant exp = 1, else = 0"""
    return np.array([1 if 'Has relevent' in str(v) else 0 for v in col], dtype=np.int32)

def encode_ordinal(col, mapping):
    """Map to ordinal values, missing=-1"""
    return np.array([mapping.get(v.strip(), -1) for v in col], dtype=np.int32)

def one_hot(values, categories=None, fit=True):
    """One-hot encode with fit/transform"""
    if fit:
        categories = sorted([v.strip() for v in set(values) if v.strip()])
    
    encoded = np.zeros((len(values), len(categories)), dtype=np.int32)
    for i, v in enumerate(values):
        if v.strip() in categories:
            encoded[i, categories.index(v.strip())] = 1
    return encoded, categories

def standardize(values, mean=None, std=None, fit=True, missing=-999.0):
    """Z-score: (x-mean)/std, missing→0"""
    valid = values != missing
    if fit:
        mean, std = np.mean(values[valid]), np.std(values[valid])
        std = std if std > 0 else 1.0
    
    result = np.zeros_like(values, dtype=np.float64)
    result[valid] = (values[valid] - mean) / std
    return result, mean, std

# ===== FEATURE ENGINEERING =====
def make_interactions(feats):
    """Create interaction features"""
    ints = {}
    if 'experience_bin' in feats and 'relevent_experience' in feats:
        exp = np.where(feats['experience_bin']==-1, 0, feats['experience_bin']).astype(float)
        ints['exp_relevance'] = exp * feats['relevent_experience']
    
    if 'company_size' in feats and 'last_new_job' in feats:
        cs = np.where(feats['company_size']==-1, 0, feats['company_size']).astype(float)
        lj = np.where(feats['last_new_job']==-1, 0, feats['last_new_job']).astype(float)
        ints['company_stability'] = cs * lj
    return ints

def make_missing_flags(feats):
    """Binary flags for missing values"""
    flags = {}
    flags['gender_missing'] = (feats['gender']==2).astype(np.int32)
    for f in ['experience_bin', 'company_size', 'last_new_job']:
        if f in feats:
            flags[f'{f}_missing'] = (feats[f]==-1).astype(np.int32)
    return flags

# ===== MAIN PIPELINE =====
def preprocess_train_dataset(filepath, save_artifacts=True, artifacts_path=None):
    """Train preprocessing pipeline"""
    print(f"Loading {filepath}...")
    data, headers = load_csv(filepath)
    print(f"  {data.shape[0]:,} samples")
    
    # Extract columns
    cols = {name: get_col(data, headers, name) for name in 
            ['city_development_index','training_hours','gender','relevent_experience',
             'experience','company_size','last_new_job','enrolled_university',
             'education_level','major_discipline','company_type','target']}
    
    # Numeric features (standardize)
    print("Standardizing numeric features...")
    cdi_num = to_numeric(cols['city_development_index'])
    hrs_num = to_numeric(cols['training_hours'])
    cdi_scaled, cdi_mean, cdi_std = standardize(cdi_num, fit=True)
    hrs_scaled, hrs_mean, hrs_std = standardize(hrs_num, fit=True)
    
    # Categorical features (encode)
    print("Encoding categorical features...")
    gender_enc = encode_gender(cols['gender'])
    relexp_enc = encode_binary(cols['relevent_experience'])
    exp_enc = encode_ordinal(cols['experience'], EXP_BINS)
    size_enc = encode_ordinal(cols['company_size'], COMP_SIZE)
    lastjob_enc = encode_ordinal(cols['last_new_job'], LAST_JOB)
    
    # One-hot features
    print("One-hot encoding...")
    uni_oh, uni_cats = one_hot(cols['enrolled_university'], fit=True)
    edu_oh, edu_cats = one_hot(cols['education_level'], fit=True)
    maj_oh, maj_cats = one_hot(cols['major_discipline'], fit=True)
    ctype_oh, ctype_cats = one_hot(cols['company_type'], fit=True)
    
    # Feature engineering
    print("Engineering features...")
    feats = {'gender': gender_enc, 'relevent_experience': relexp_enc, 
             'experience_bin': exp_enc, 'company_size': size_enc, 'last_new_job': lastjob_enc}
    interactions = make_interactions(feats)
    missing_flags = make_missing_flags(feats)
    
    # Build feature matrix
    print("Building feature matrix...")
    blocks, names = [], []
    
    # Add numeric (2)
    blocks.extend([cdi_scaled.reshape(-1,1), hrs_scaled.reshape(-1,1)])
    names.extend(['cdi_scaled', 'hours_scaled'])
    
    # Add categorical (5)
    blocks.extend([gender_enc.reshape(-1,1), relexp_enc.reshape(-1,1), exp_enc.reshape(-1,1),
                   size_enc.reshape(-1,1), lastjob_enc.reshape(-1,1)])
    names.extend(['gender', 'rel_exp', 'exp_bin', 'comp_size', 'last_job'])
    
    # Add one-hot (~20)
    for oh, cats, prefix in [(uni_oh,uni_cats,'uni'), (edu_oh,edu_cats,'edu'),
                             (maj_oh,maj_cats,'maj'), (ctype_oh,ctype_cats,'ctype')]:
        if oh.shape[1] > 0:
            blocks.append(oh)
            names.extend([f'{prefix}_{c}' for c in cats])
    
    # Add engineered (6)
    for name, vals in interactions.items():
        blocks.append(vals.reshape(-1,1))
        names.append(name)
    for name, vals in missing_flags.items():
        blocks.append(vals.reshape(-1,1))
        names.append(name)
    
    X = np.concatenate(blocks, axis=1).astype(np.float64)
    y = to_numeric(cols['target'], missing=0.0)
    
    # Save artifacts
    artifacts = {
        'numeric_stats': {'city_development_index': {'mean': float(cdi_mean), 'std': float(cdi_std)},
                         'training_hours': {'mean': float(hrs_mean), 'std': float(hrs_std)}},
        'categorical_mappings': {'enrolled_university': uni_cats, 'education_level': edu_cats, 
                                'major_discipline': maj_cats, 'company_type': ctype_cats},
        'feature_names': names
    }
    
    if save_artifacts and artifacts_path:
        with open(artifacts_path, 'w') as f:
            json.dump(artifacts, f, indent=2)
        print(f"Artifacts → {artifacts_path}")
    
    print(f"Done! X: {X.shape}, y: {y.shape}, features: {len(names)}")
    return {'X': X, 'y': y, 'feature_names': names, 'artifacts': artifacts}

def preprocess_test_dataset(filepath, artifacts_path):
    """Test preprocessing using train artifacts"""
    print(f"Loading {filepath}...")
    data, headers = load_csv(filepath)
    print(f"  {data.shape[0]:,} samples")
    
    # Load artifacts
    with open(artifacts_path, 'r') as f:
        arts = json.load(f)
    
    enrollee_ids = get_col(data, headers, 'enrollee_id')
    
    # Extract columns
    cols = {name: get_col(data, headers, name) for name in 
            ['city_development_index','training_hours','gender','relevent_experience',
             'experience','company_size','last_new_job','enrolled_university',
             'education_level','major_discipline','company_type']}
    
    # Numeric (use train stats)
    print("Standardizing with train stats...")
    cdi_scaled, _, _ = standardize(to_numeric(cols['city_development_index']),
                                    mean=arts['numeric_stats']['city_development_index']['mean'],
                                    std=arts['numeric_stats']['city_development_index']['std'], fit=False)
    hrs_scaled, _, _ = standardize(to_numeric(cols['training_hours']),
                                    mean=arts['numeric_stats']['training_hours']['mean'],
                                    std=arts['numeric_stats']['training_hours']['std'], fit=False)
    
    # Categorical
    print("Encoding...")
    gender_enc = encode_gender(cols['gender'])
    relexp_enc = encode_binary(cols['relevent_experience'])
    exp_enc = encode_ordinal(cols['experience'], EXP_BINS)
    size_enc = encode_ordinal(cols['company_size'], COMP_SIZE)
    lastjob_enc = encode_ordinal(cols['last_new_job'], LAST_JOB)
    
    # One-hot (use train categories)
    print("One-hot encoding...")
    uni_oh, _ = one_hot(cols['enrolled_university'], arts['categorical_mappings']['enrolled_university'], fit=False)
    edu_oh, _ = one_hot(cols['education_level'], arts['categorical_mappings']['education_level'], fit=False)
    maj_oh, _ = one_hot(cols['major_discipline'], arts['categorical_mappings']['major_discipline'], fit=False)
    ctype_oh, _ = one_hot(cols['company_type'], arts['categorical_mappings']['company_type'], fit=False)
    
    # Engineering
    print("Engineering...")
    feats = {'gender': gender_enc, 'relevent_experience': relexp_enc,
             'experience_bin': exp_enc, 'company_size': size_enc, 'last_new_job': lastjob_enc}
    interactions = make_interactions(feats)
    missing_flags = make_missing_flags(feats)
    
    # Build X (same order as train)
    blocks = [cdi_scaled.reshape(-1,1), hrs_scaled.reshape(-1,1),
              gender_enc.reshape(-1,1), relexp_enc.reshape(-1,1), exp_enc.reshape(-1,1),
              size_enc.reshape(-1,1), lastjob_enc.reshape(-1,1)]
    
    for oh in [uni_oh, edu_oh, maj_oh, ctype_oh]:
        if oh.shape[1] > 0: blocks.append(oh)
    
    for vals in list(interactions.values()) + list(missing_flags.values()):
        blocks.append(vals.reshape(-1,1))
    
    X = np.concatenate(blocks, axis=1).astype(np.float64)
    
    print(f"Done! X: {X.shape}, expected: {len(arts['feature_names'])}")
    if X.shape[1] != len(arts['feature_names']):
        print(f"Feature mismatch!")
    
    return {'X': X, 'feature_names': arts['feature_names'], 'enrollee_ids': enrollee_ids}

def save_processed_data(X, y, feature_names, filepath, enrollee_ids=None):
    """Save processed data to CSV"""
    header = (['enrollee_id'] if enrollee_ids is not None else []) + feature_names + (['target'] if y is not None else [])
    
    data_list = []
    if enrollee_ids is not None:
        data_list.append(enrollee_ids.reshape(-1,1))
    data_list.append(X)
    if y is not None:
        data_list.append(y.reshape(-1,1))
    
    combined = np.concatenate(data_list, axis=1)
    np.savetxt(filepath, combined, delimiter=',', header=','.join(header), comments='', fmt='%s')
    print(f"Saved {filepath} ({combined.shape})")
