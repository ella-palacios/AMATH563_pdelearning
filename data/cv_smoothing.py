test_results,y_samples = pendulum_train_data(I = 1)

u1 = train_results[:,0,:]
u2 = train_results[:,1,:]
t = np.linspace(0,1,1000)

folds = np.floor(np.random.rand(1000)*10)

def crossval_lin(c,a,lam):
    MSE = 0
    K = linmatrix_pend(t,c,a)
    for i in range(10):
        train = np.where(folds!=i)[0]
        train_u1 = np.reshape(u1[:,train],[1,len(train)])
        train_u2 = np.reshape(u2[:,train],[1,len(train)])
        train_t = t[train]
        train_K = np.copy(K[train,:])
        train_K = np.copy(train_K[:,train])
        
        test = np.where(folds==i)[0]
        test_u1 = np.reshape(u1[:,test],[1,len(test)])
        test_u2 = np.reshape(u2[:,test],[1,len(test)])
        test_t = t[test]
        
        
        for k in range(1):
            regvec1 = np.linalg.solve(train_K + lam*np.eye(len(train),len(train)),train_u1[k,:])
            regvec2 = np.linalg.solve(train_K + lam*np.eye(len(train),len(train)),train_u2[k,:])
        
            predictions1 = np.zeros(len(test))
            predictions2 = np.zeros(len(test))
            for i in range(len(test)):
                kervec1 = np.zeros(len(train))
                kervec2 = np.zeros(len(train))
                for j in range(len(train)):
                    kervec1[j] = (c + np.inner(test_u1[k,i],train_u1[k,j]))**a
                    kervec2[j] = (c + np.inner(test_u2[k,i],train_u2[k,j]))**a
                predictions1[i] = np.inner(kervec1,regvec1)
                predictions2[i] = np.inner(kervec2,regvec2)
        
            MSE += (1/20)*np.inner(predictions1-test_u1[k,:],predictions1-test_u1[k,:])/np.inner(test_u1[k,:],test_u1[k,:])
            MSE += (1/20)*np.inner(predictions2-test_u2[k,:],predictions2-test_u2[k,:])/np.inner(test_u2[k,:],test_u2[k,:])
    return MSE


results,y_samples = pendulum_train_data(I = 1)


u1 = results[:,0,:]
u2 = results[:,1,:]
t = np.linspace(0,1,1000)

folds = np.floor(np.random.rand(1000)*10)

def crossval_rbf(l,lam):
    MSE = 0
    K = kermatrix_pend(t,l)
    for i in range(10):
        train = np.where(folds!=i)[0]
        train_u1 = np.reshape(u1[:,train],[1,len(train)])
        train_u2 = np.reshape(u2[:,train],[1,len(train)])
        train_t = t[train]
        train_K = np.copy(K[train,:])
        train_K = np.copy(train_K[:,train])
        
        test = np.where(folds==i)[0]
        test_u1 = np.reshape(u1[:,test],[1,len(test)])
        test_u2 = np.reshape(u2[:,test],[1,len(test)])
        test_t = t[test]
        
        
        for k in range(1):
            regvec1 = np.linalg.solve(train_K + lam*np.eye(len(train),len(train)),train_u1[k,:])
            regvec2 = np.linalg.solve(train_K + lam*np.eye(len(train),len(train)),train_u2[k,:])
        
            predictions1 = np.zeros(len(test))
            predictions2 = np.zeros(len(test))
            for i in range(len(test)):
                kervec1 = np.zeros(len(train))
                kervec2 = np.zeros(len(train))
                for j in range(len(train)):
                    kervec1[j] = np.exp((-1/(2*(l**2)))*((test_u1[k,i]-train_u1[k,j])**2))
                    kervec2[j] = np.exp((-1/(2*(l**2)))*((test_u2[k,i]-train_u2[k,j])**2))
                predictions1[i] = np.inner(kervec1,regvec1)
                predictions2[i] = np.inner(kervec2,regvec2)
        
            MSE += (1/20)*np.inner(predictions1-test_u1[k,:],predictions1-test_u1[k,:])/np.inner(test_u1[k,:],test_u1[k,:])
            MSE += (1/20)*np.inner(predictions2-test_u2[k,:],predictions2-test_u2[k,:])/np.inner(test_u2[k,:],test_u2[k,:])
    return MSE





# --- using the crossvalidation

# test_c = np.array([.005,.01,.2])
# test_a = np.array([2,3,4,5])
# test_lam = np.array([.1,.25,.4])
# lin_test_MSE = np.zeros([3,4,3])

# for i in range(3):
#     for j in range(4):
#         for k in range(3):
#             lin_test_MSE[i,j,k] = crossval_lin(c=test_c[i],a = test_a[j],lam=test_lam[k])
#     print("done with",(12*(i+1)),"!!!!!")