import numpy as np
import pytest
from network import NeuralNetwork

@pytest.mark.parametrize('activation',['sigmoid','relu','leaky_relu','tanh'])
@pytest.mark.parametrize('mode',['classification','regression'])
def test_every_parameter_gradient_matches_finite_differences(activation,mode):
    rng=np.random.default_rng(7);model=NeuralNetwork([3,4,2],mode=mode,activation=activation)
    x=rng.normal(size=(5,3));y=np.eye(2)[[0,1,0,1,1]] if mode=='classification' else rng.normal(size=(5,2))
    acts=model.forward_pass(x);deltas=model.backward_pass(y,acts)
    inputs=[x]+acts[:-1];gradw=[a.T@d for a,d in zip(inputs,deltas)];gradb=[d.sum(axis=0) for d in deltas]
    loss=model.cross_entropy_loss if mode=='classification' else model.mean_squared_loss
    for arrays,grads in [(model.weights_,gradw),(model.biases_,gradb)]:
        for array,expected in zip(arrays,grads):
            numerical=np.zeros_like(array)
            for index in np.ndindex(array.shape):
                old=array[index];array[index]=old+1e-6;plus=loss(model.predict(x),y)
                array[index]=old-1e-6;minus=loss(model.predict(x),y);array[index]=old
                numerical[index]=(plus-minus)/2e-6
            np.testing.assert_allclose(expected,numerical,atol=1e-7,rtol=1e-5)

@pytest.mark.parametrize('optimizer',['sgd','minibatch','fullbatch'])
def test_all_optimizers_learn_and_restore_best_validation_checkpoint(optimizer):
    rng=np.random.default_rng(9);x=np.r_[rng.normal(-2,.2,(20,2)),rng.normal(2,.2,(20,2))];y=np.r_[np.zeros(20,int),np.ones(20,int)]
    model=NeuralNetwork([2,6,2],optimizer=optimizer,activation='tanh',batch_size=8)
    history,val=model.fit(x,y,x,y,epochs=50,lr=.05)
    assert len(history)==len(val)==50 and model.evaluate(x,y)[1]>.98
    assert abs(model.evaluate(x,y)[0]-min(val))<1e-12
    assert model.selected_epoch==int(np.argmin(val))+1

def test_multioutput_regression_learns_linear_map():
    rng=np.random.default_rng(8);x=rng.normal(size=(100,2));y=x@np.array([[2.,-1.],[1.,3.]])+.5
    model=NeuralNetwork([2,2],mode='regression');model.fit(x,y,x,y,epochs=200,lr=.1)
    assert model.evaluate(x,y)<1e-10

def test_integer_and_one_hot_targets_have_same_training_trajectory():
    x=np.array([[-2.,-1.],[-1.,-2.],[2.,1.],[1.,2.]]);y=np.array([0,0,1,1])
    a=NeuralNetwork([2,4,2]);b=NeuralNetwork([2,4,2])
    a.fit(x,y,x,y,10,.1);b.fit(x,np.eye(2)[y],x,np.eye(2)[y],10,.1)
    np.testing.assert_array_equal(a.predict(x),b.predict(x))

def test_invalid_settings_features_targets_and_sigmoid_extremes():
    for args in [{'nodes_per_layer':[2,0,1]},{'nodes_per_layer':[2,2.5,1]}, {'nodes_per_layer':[2,1],'batch_size':0},
                 {'nodes_per_layer':[2,1],'optimizer':'bad'}]:
        with pytest.raises(ValueError):NeuralNetwork(**args)
    model=NeuralNetwork([2,3,2]);x=np.ones((3,2));y=np.array([0,1,0])
    for targets in [np.array([0,2,0]),np.array([0.,1.,0.]),np.ones((3,2))]:
        with pytest.raises(ValueError):model.fit(x,targets,x,y,1)
    with pytest.raises(ValueError):model.fit(x,y,x,y,0)
    with pytest.raises(ValueError):model.predict(np.full((1,2),np.nan))
    np.testing.assert_array_equal(model.sigmoid(np.array([-10000.,10000.])),[0.,1.])
