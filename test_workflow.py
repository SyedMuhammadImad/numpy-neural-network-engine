import numpy as np
from sklearn.preprocessing import StandardScaler
from network import NeuralNetwork
from experiment import split,save_model,load_model,predict

def test_train_only_scale_and_disjoint_splits():
    rng=np.random.default_rng(42);x=rng.normal(size=(120,2));y=np.arange(120)%2
    scaler,parts=split(x,y)
    assert [len(p[2]) for p in parts]==[72,24,24]
    assert all(set(parts[i][2]).isdisjoint(parts[j][2]) for i in range(3) for j in range(i))
    np.testing.assert_allclose(scaler.mean_,x[parts[0][2]].mean(axis=0))

def test_scale_correct_regression_checkpoint_prediction(tmp_path):
    rng=np.random.default_rng(8);raw=rng.normal(100,20,(100,2));y=(raw[:,0]*2+raw[:,1]*3)[:,None]
    scaler=StandardScaler().fit(raw);mean=y.mean();scale=y.std()
    model=NeuralNetwork([2,1],mode='regression');model.fit(scaler.transform(raw),(y-mean)/scale,scaler.transform(raw),(y-mean)/scale,150,.1)
    save_model(tmp_path,'regression',model,scaler,float(mean),float(scale))
    np.testing.assert_array_equal(predict(tmp_path,'regression',raw),model.predict(scaler.transform(raw))*scale+mean)
    assert np.mean((predict(tmp_path,'regression',raw)-y)**2)<1e-9
