"""Fresh manual-network classification/regression experiments on disjoint splits."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,mean_squared_error,r2_score
from network import NeuralNetwork

def save_model(folder,name,model,scaler,target_mean=0.,target_scale=1.):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    arrays={f'w{i}':w for i,w in enumerate(model.weights_)}
    arrays.update({f'b{i}':b for i,b in enumerate(model.biases_)})
    arrays.update(mean=scaler.mean_,scale=scaler.scale_,target_mean=np.array(target_mean),target_scale=np.array(target_scale))
    np.savez(folder/(name+'.npz'),**arrays)
    settings={'nodes':model.nodes_per_layer,'mode':model.mode,'activation':model.activation_strat,'optimizer':model.optimizer,'batch_size':model.batch_size}
    (folder/(name+'.json')).write_text(json.dumps(settings),encoding='utf-8')

def load_model(folder,name):
    folder=Path(folder);s=json.loads((folder/(name+'.json')).read_text(encoding='utf-8'))
    model=NeuralNetwork(s['nodes'],s['mode'],s['optimizer'],s['activation'],s['batch_size'])
    with np.load(folder/(name+'.npz'),allow_pickle=False) as a:
        model.weights_=[a[f'w{i}'].copy() for i in range(len(s['nodes'])-1)]
        model.biases_=[a[f'b{i}'].copy() for i in range(len(s['nodes'])-1)]
        scaling={k:a[k].copy() for k in ['mean','scale','target_mean','target_scale']}
    return model,scaling

def predict(folder,name,x):
    model,s=load_model(folder,name)
    value=model.predict((np.asarray(x,dtype=np.float64)-s['mean'])/s['scale'])
    return value if model.mode=='classification' else value*s['target_scale']+s['target_mean']

def classification_score(y,probs):
    pred=probs.argmax(axis=1)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),
            'confusion_matrix':confusion_matrix(y,pred,labels=np.arange(probs.shape[1])).tolist()}

def split(x,y):
    ti,ei=train_test_split(np.arange(len(x)),test_size=.2,random_state=42,stratify=y if y.ndim==1 else None)
    ti,vi=train_test_split(ti,test_size=.25,random_state=42,stratify=y[ti] if y.ndim==1 else None)
    scaler=StandardScaler().fit(x[ti])
    return scaler,[(scaler.transform(x[i]),y[i],i) for i in (ti,vi,ei)]

def run_classification(output):
    x,y=make_moons(n_samples=1800,noise=.2,random_state=42)
    scaler,parts=split(x,y);(tx,ty,ti),(vx,vy,vi),(ex,ey,ei)=parts
    configs=[('fullbatch_sigmoid',[2,30,30,2],'fullbatch','sigmoid',.1,500,64),
             ('sgd_tanh',[2,20,20,2],'sgd','tanh',.01,200,1),
             ('minibatch_relu',[2,40,40,2],'minibatch','relu',.01,300,32),
             ('minibatch_leaky_relu',[2,60,40,20,2],'minibatch','leaky_relu',.05,400,64),
             ('minibatch_relu_high_lr',[2,50,50,2],'minibatch','relu',.1,400,32)]
    result={'dataset':'synthetic make_moons; 1800 samples, noise .2, seed 42','split_counts':{'train':len(tx),'validation':len(vx),'test':len(ex)},'models':{}}
    for name,nodes,opt,act,lr,epochs,batch in configs:
        model=NeuralNetwork(nodes,optimizer=opt,activation=act,batch_size=batch)
        history,val=model.fit(tx,ty,vx,vy,epochs,lr)
        vp,ep=model.predict(vx),model.predict(ex)
        save_model(output,name,model,scaler)
        np.testing.assert_allclose(ep,predict(output,name,x[ei]),atol=0,rtol=0)
        result['models'][name]={'settings':{'nodes':nodes,'optimizer':opt,'activation':act,'learning_rate':lr,'epochs':epochs,'batch_size':batch},
            'selected_epoch':model.selected_epoch,'validation_cross_entropy':float(min(val)),
            'validation':classification_score(vy,vp),'test':classification_score(ey,ep),'checkpoint_reload_identical':True,
            'training_history':history,'validation_history':val}
        print('moons '+name+': '+str(result['models'][name]['test']['accuracy']),flush=True)
    return result

def run_regression(output):
    rng=np.random.default_rng(42);x=rng.uniform(-200,200,(1800,2));a,b=x[:,0:1],x[:,1:2]
    y=-(b+47)*np.sin(np.sqrt(np.abs(b+a/2+47)))-a*np.sin(np.sqrt(np.abs(a-(b+47))))+.1*rng.normal(size=(1800,1))
    scaler,parts=split(x,y);(tx,ty,ti),(vx,vy,vi),(ex,ey,ei)=parts
    mean,scale=ty.mean(),ty.std();nty=(ty-mean)/scale;nvy=(vy-mean)/scale
    configs=[('eggholder_tanh',[2,64,64,1],'tanh',.005),('eggholder_relu',[2,64,64,32,1],'relu',.0001),('eggholder_sigmoid',[2,50,50,1],'sigmoid',.01)]
    result={'dataset':'synthetic noisy Eggholder, 1800 samples, seed 42','split_counts':{'train':len(tx),'validation':len(vx),'test':len(ex)},'models':{}}
    for name,nodes,act,lr in configs:
        model=NeuralNetwork(nodes,mode='regression',optimizer='minibatch',activation=act,batch_size=32)
        history,val=model.fit(tx,nty,vx,nvy,500,lr)
        ep=model.predict(ex)*scale+mean
        save_model(output,name,model,scaler,float(mean),float(scale))
        np.testing.assert_allclose(ep,predict(output,name,x[ei]),atol=0,rtol=0)
        result['models'][name]={'settings':{'nodes':nodes,'activation':act,'learning_rate':lr,'epochs':500},
            'selected_epoch':model.selected_epoch,'validation_normalized_mse':float(min(val)),
            'test':{'mse_original_units':float(mean_squared_error(ey,ep)),'r2':float(r2_score(ey,ep)),
                    'training_mean_baseline_mse':float(mean_squared_error(ey,np.full_like(ey,mean)))},
            'checkpoint_reload_identical':True,'training_history':history,'validation_history':val}
        print('regression '+name+': R2 '+str(result['models'][name]['test']['r2']),flush=True)
    return result

def run_fashion(data_dir,output,download=False):
    from torchvision.datasets import FashionMNIST
    a=FashionMNIST(data_dir,train=True,download=download);b=FashionMNIST(data_dir,train=False,download=download)
    rng=np.random.default_rng(42);selected=rng.choice(len(a),20000,replace=False)
    raw=a.data.numpy()[selected].reshape(-1,784).astype(np.float64)/255
    labels=a.targets.numpy()[selected];ti,vi=train_test_split(np.arange(len(raw)),test_size=.1,stratify=labels,random_state=42)
    scaler=StandardScaler().fit(raw[ti])
    tx,vx=scaler.transform(raw[ti]),scaler.transform(raw[vi])
    ex=b.data.numpy().reshape(-1,784).astype(np.float64)/255;ey=b.targets.numpy()
    model=NeuralNetwork([784,20,20,10],activation='relu',optimizer='minibatch',batch_size=64)
    history,val=model.fit(tx,labels[ti],vx,labels[vi],30,.01)
    ep=model.predict(scaler.transform(ex));name='fashion_manual_mlp'
    save_model(output,name,model,scaler);np.testing.assert_allclose(ep,predict(output,name,ex),atol=0,rtol=0)
    result={'dataset':'Fashion-MNIST official partitions; 20000-sample training subset, 10000 official tests',
        'split_counts':{'train':len(ti),'validation':len(vi),'test':len(ey)},
        'data_fingerprint':hashlib.sha256(raw.tobytes()+labels.tobytes()).hexdigest(),
        'models':{name:{'settings':{'nodes':[784,20,20,10],'epochs':30,'learning_rate':.01,'optimizer':'minibatch','activation':'relu'},
            'selected_epoch':model.selected_epoch,'validation_cross_entropy':float(min(val)),
            'test':classification_score(ey,ep),'checkpoint_reload_identical':True,'training_history':history,'validation_history':val}}}
    print('fashion '+str(result['models'][name]['test']['accuracy']),flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--download',action='store_true')
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    result={'seed':42,'selection':'lowest validation loss, held-out tests never select checkpoints',
        'moons_classification':run_classification(out),'eggholder_regression':run_regression(out),
        'fashion_classification':run_fashion(args.data_dir,out,args.download)}
    (out/'metrics.json').write_text(json.dumps(result,indent=2),encoding='utf-8')

if __name__=='__main__':main()
