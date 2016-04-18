
trainlabel=dlmread('/pic/train-idx1.txt');
testlabel=dlmread('/pic/test-idx1.txt');
savedirectory='/pic/train-idx3/';
savedirectory2='/pic/tk10-idx3/';

trans=[];
emis=[];
prior=[];
for j=1:9
    path1=[];
    for i=1:60000
        if trainlabel(i)==j
            path1=[path1,i];
        end 
    end
    num=randperm(length(path1),100);
    data=[];
    for k=1:100
        savepath = fullfile(savedirectory,['TestImage_' num2str(path1(num(k)),'%d') '.bmp']);
        A=imread(savepath);
        B = [];
        for m = 0:13
            for n = 0:13
                x = m * 2 + 1;
                y = n * 2 + 1;
                B(m+1, n+1) = uint8(sum(sum(A(x:(x+1), y:(y+1)))) / 4);
            end
        end
        data1=[];
        for p=1:14
            for q =1:14
                if B(p,q)<86
                    data1(p,q)=1;
                elseif B(p,q)<170
                    data1(p,q)=2;
                else
                    data1(p,q)=3;
                end
            end
        end
        data=[data;reshape(data1,1,14*14)];
    end
    O = 3;
    Q = 5;
    T=size(data,2);
    X=size(data,1);
    transmat0=zeros(Q,Q);
    transmat0(Q,Q)=1;
    for a = 1:(Q-1)
        for b = a:(Q-1)
            if b == a
                transmat0(a,b) = 0.5;
            elseif b== a + 1
                transmat0(a,b) = 0.5;
            end
        end
    end
    prior0 = normalise(rand(Q,1));
    obsmat0 = mk_stochastic(rand(Q,O)); 
    [LL, prior1, transmat1, obsmat1] = dhmm_em(data, prior0, transmat0, obsmat0, 'max_iter', 5);
    B = multinomial_prob(data, obsmat1);
    [path] = viterbi_path(prior1, transmat1, B);  
    trans=[trans;transmat1];
    emis=[emis;obsmat1];
    prior=[prior;prior1];
       
end
count=0;
for i=1:1000
    savepath = fullfile(savedirectory2,['TestImage_' num2str(i,'%d') '.bmp']);
    A=imread(savepath);
    B = [];
    for m = 0:13
        for n = 0:13
            x = m * 2 + 1;
            y = n * 2 + 1;
            B(m+1, n+1) = uint8(sum(sum(A(x:(x+1), y:(y+1)))) / 4);
        end
    end
    data1=[];
    for p=1:14
        for q =1:14
            if B(p,q)<86
                data1(p,q)=1;
            elseif B(p,q)<170
                data1(p,q)=2;
            else
                data1(p,q)=3;
            end
        end
    end
    data=reshape(data1,1,14*14);
    loglik=[];
    class=1;
    for j=1:9
        loglik(j) = dhmm_logprob(data, prior((j-1)*Q+1:j*Q,1), trans((j-1)*Q+1:j*Q,1:Q), emis((j-1)*Q+1:j*Q,1:O));   
    end
    for j=2:9
        if loglik(j)>loglik(class)
            class=j;
        end
    end
    if class==testlabel(i)
        count=count+1;
    end
    
end



  
    
        




