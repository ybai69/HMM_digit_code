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
        for m=1:28
            for n=1:28
                if A(m,n)<128
                    A(m,n)=1;
                else A(m,n)=0;
                end
            end
        end
        [m,n]=size(A);
        temp1=zeros(1,m);
        temp2=zeros(1,n);
        for k=1:m
            temp1(k)=sum(A(k,:));
        end
        for k=1:n
            temp2(k)=sum(A(:,k));
        end
        temp=[temp1,temp2];
        data1=temp+1;
        
        data=[data;data1];
    end
    O = 29;
    Q = 15;
    T=size(data,2);
    X=size(data,1);
    transmat0=zeros(Q,Q);
    transmat0(Q,Q)=1;
    transmat0(Q-1,Q-1:Q)=0.5;
    for a = 1:(Q-2)
        for b = a:Q
            if b == a
                transmat0(a,b) = 1/3;
            elseif b== a + 1
                transmat0(a,b) = 1/3;
            elseif b==a+2
                transmat0(a,b)=1/3;
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
    for m=1:28
        for n=1:28
            if A(m,n)<128
                A(m,n)=1;
            else A(m,n)=0;
            end
        end
    end
    [m,n]=size(A);
    temp1=zeros(1,m);
    temp2=zeros(1,n);
    for k=1:m
        temp1(k)=sum(A(k,:));
    end
    for k=1:n
        temp2(k)=sum(A(:,k));
    end
    temp=[temp1,temp2]
    data=temp+1;
        
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



  
    
        



