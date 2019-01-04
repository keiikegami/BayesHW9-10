#include<oxstd.h>
#include<oxprob.h>
#include<oxfloat.h>
#include<oxdraw.h>
#import<maximize>	   

/*
** FFBSamp implements Backward "sampling" for a DLM 
**
** [Input]
** nsim: number of RVs to sample  
** vy: Tx1 time series (univariate)
** mF: Txp predictors. Each row means F_t'
** vV: Tx1 obs variances V_t
** amG: T arrays of pxp matrices G_t (state regressors).
** amW: T arrays of pxp matrices W_t (state variance).
** vm0 and mC0: state at 0 follows N(vm0,mC0)
**
** [Output]	Samples from retrospective posterior
** amb: T array of nsim x p matrix.
** amb[t] is the matrix of theta_t, each row is a sampled vector theta_t'
**
*/
FFBS(const nsim, const vy, const mF, const vV, const amG, const amW, const vm0, const mC0){

decl nT,t,p;
decl vm,mC,dq,mA;
decl ma,amR,mm,amC,mB;
decl mb,amb,mH;

nT = sizer(mF); p = sizec(mF);

ma = mm = zeros(nT,p);
amR = amC = new array[nT];

vm = vm0; mC = mC0;

amb = new array[nT]; // Sampled 
					 // Array of nsim x np
// FF
for(t=0;t<nT;t++){

ma[t][] = vm';
amR[t] = amG[t]*mC*(amG[t]') + amW[t];

dq = mF[t][]*amR[t]*(mF[t][]') + vV[t];
mA = amR[t]*(mF[t][]')/dq;

vm = ma[t][]' + mA*(vy[t]-mF[t][]*vm);
mC = amR[t] - dq*mA*(mA');

mm[t][] = vm';
amC[t] = mC;
			
}

// B-Sampling

mb = vm' + rann(nsim,p) *  choleski(mC)';
amb[nT-1] = mb;
			   
	for(t=nT-2;t>=0;t--){

	mB = amC[t]*(amG[t+1]')*invert(amR[t+1]); 
	mH = choleski( amC[t] - mB*amR[t+1]*(mB') );                            		  
		   
	mb = mm[t][] + (mb-ma[t+1][])*(mB') + rann(nsim,p)*(mH');                            

	amb[t] = mb;
	
	}

								
return amb;

}




main(){

decl mdata,vy,mF,vV,amG,amW,vm0,mC0;
decl nT,np,t,i,mm,amC,amb,vh_true,mh;
decl dmu, dphi, dsig2;
decl nsim;

// Data load
vy = loadmat("sv_simdata.csv");	   
nT = sizec(vy);

// MC size
nsim = 5000;

// Fix parameters
vm0=0;   // initial state mean
mC0=0.1;  // initial state variance
dmu = 0; dphi = 0.95; dsig2 = 0.1;
amG = amW = new array[nT];
amG[] = dphi; // AR phi 
amW[] = dsig2;  // AR sig^2

// Normal approx log-chi^2 = N(-1.27,22.2^2)
vy = log(vy.^2);
vy += 1.27 - dmu;
vV = (2.22^2)*ones(nT,1); // obs variance ... let's use the true value
mF = ones(nT,1);			

// FF-BSampling
amb = FFBS(nsim,vy,mF,vV,amG,amW,vm0,mC0);

// Summarize the results into quantiles
mh = zeros(3,nT);
for(i=0;i<nT;i++){
mh[0][i] = quantilec(amb[i]+dmu,0.025);
mh[1][i] = meanc(amb[i]+dmu);
mh[2][i] = quantilec(amb[i]+dmu,0.975);
}



// True value
vh_true = loadmat("sv_truevol.csv");
					  
// Draw the results
	DrawT(0, mh[0][],1,1,1);		DrawAdjust(ADJ_COLOR,3,10); 	
	DrawT(0, mh[1][],1,1,1);		DrawAdjust(ADJ_COLOR,3,12); 	DrawTitle(0, sprint("Post mean \& 95\% CIs of $h_t$"));
	DrawT(0, mh[2][],1,1,1);		DrawAdjust(ADJ_COLOR,3,10);
	DrawT(0, vh_true,1,1,1);		DrawAdjust(ADJ_COLOR,2,12); 	DrawTitle(0, sprint("True log-volatility"));
//	DrawT(0, vy,1,1,1);				DrawAdjust(ADJ_COLOR,1,12); 	DrawTitle(0, sprint("Data"));
SaveDrawWindow(sprint("post_sv.eps"));
CloseDrawWindow();
										 
	
	
}