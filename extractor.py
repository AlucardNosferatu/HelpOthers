import os

import pandas as pd
from radiomics import featureextractor

imageFile = 'data/featureExtraction/brain1/brain1_image.nrrd'
maskFile = 'data/featureExtraction/brain1/brain1_label.nrrd'

# extractor = featureextractor.RadiomicsFeatureExtractor()
# featureVector = extractor.execute(imageFile, maskFile)
# for featureName in featureVector.keys():
#     print("%s: %s" % (featureName, featureVector[featureName]))

# extractor.disableAllFeatures()
# extractor.enableFeaturesByName(firstorder=['Mean', 'Skewness'])

# featureVector = extractor.execute(imageFile, maskFile)
# for featureName in featureVector.keys():
#     print("%s: %s" % (featureName, featureVector[featureName]))

# extractor.disableAllFeatures()
# extractor.enableFeatureClassByName('glcm')

# featureVector = extractor.execute(imageFile, maskFile)
# for featureName in featureVector.keys():
#     print("%s: %s" % (featureName, featureVector[featureName]))

# extractor.enableImageTypes(Original={}, LoG={}, Wavelet={})

# featureVector = extractor.execute(imageFile, maskFile)
# for featureName in featureVector.keys():
#     print("%s: %s" % (featureName, featureVector[featureName]))

basePath = 'data/featureExtraction'
folders = os.listdir(basePath)
print(folders)

df = pd.DataFrame()
for folder in folders:
    if os.path.isdir(os.path.join(basePath, folder)):
        files = os.listdir(os.path.join(basePath, folder))
        #     print(files)
        for file in files:
            if file.endswith('image.nrrd'):
                imageFile = os.path.join(basePath, folder, file)
            if file.endswith('label.nrrd'):
                maskFile = os.path.join(basePath, folder, file)
        #     print(imageFile, maskFile)
        extractor = featureextractor.RadiomicsFeatureExtractor()
        featureVector = extractor.execute(imageFile, maskFile)
        df_new = pd.DataFrame.from_dict(featureVector.values()).T
        df_new.columns = featureVector.keys()
        df = pd.concat([df, df_new])
    df.to_excel(os.path.join(basePath, 'results.xlsx'))
