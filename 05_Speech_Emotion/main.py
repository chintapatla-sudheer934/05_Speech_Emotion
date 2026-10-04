from pathlib import Path
import numpy as np, librosa
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report

DATA=Path("dataset")
def features(path):
    y,sr=librosa.load(path,duration=3,offset=0.5)
    return np.mean(librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40).T,axis=0)
def main():
    X=[]; labels=[]
    for folder in DATA.iterdir():
        if folder.is_dir():
            for f in folder.glob("*.wav"):
                try: X.append(features(f)); labels.append(folder.name)
                except Exception as e: print("Skipped",f,e)
    if len(set(labels))<2 or len(labels)<4:
        print("Place labeled WAVs in dataset/<emotion_name>/ folders."); return
    Xtr,Xte,ytr,yte=train_test_split(X,labels,test_size=.25,random_state=42,stratify=labels)
    model=make_pipeline(StandardScaler(),SVC())
    model.fit(Xtr,ytr)
    print(classification_report(yte,model.predict(Xte)))
    path=input("WAV file to predict: ").strip().strip('"')
    if path: print("Predicted emotion:",model.predict([features(path)])[0])
if __name__=="__main__": main()
