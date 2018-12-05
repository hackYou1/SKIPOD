#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define  Max(a,b) ((a)>(b)?(a):(b))
#define  Min(a,b) ((a)<(b)?(a):(b))

#define  N   (2048+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
int nThreads = 4;

double A [N][N];

void relax();
void init();
void verify(); 

int main(int an, char **as)
{
	int it;
    if (omp_get_num_threads() > 4) {


    }
//	double time = omp_get_wtime();
	omp_set_num_threads(1);
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		printf( "it=%4i   eps=%f\n", it,eps);
		if (eps < maxeps) break;
	}

	verify();
//	printf("time = %f\n", omp_get_wtime()-time);
	return 0;
}


void init()
{ 
	for(i=0; i<=N-1; i++)
	#pragma omp parallel for private(j),  shared(A, i)
	for(j=0; j<=N-1; j++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1)
		A[i][j]= 0.;
		else A[i][j]= ( 1. + i + j ) ;
	}
} 


void relax()
{
	int iam, limit;
	int numt = omp_get_num_threads();
	int* isync = (int*) malloc(numt*sizeof(int));
    double maxii = 0;
    double maxii_shared = 0;
#pragma omp parallel private(iam, limit, i), shared(A,numt, maxii_shared) firstprivate(maxii)
{	
	iam = omp_get_thread_num();
	limit = Min(numt-1, N-2);
	isync[iam] = 0;
    
    omp_lock_t lock;
    omp_init_lock(&lock);
	#pragma omp barrier
	for (i=1; i<=N-2; i++)
	{
		if ((iam>0) && (iam<=limit))
		{
			for ( ; isync[iam-1]==0; )
			{
				#pragma omp flush(isync)
			}
			isync[iam-1]=0;
			#pragma omp flush(isync)
		}
		#pragma omp for private(j) nowait
		for(j=1; j<=N-2; j++)
		{

			double e;
			e=A[i][j];
			A[i][j]=(A[i-1][j]+A[i+1][j]+A[i][j-1]+A[i][j+1])/4.;
            //omp_set_lock(&lock);
			maxii=Max(maxii, fabs(e-A[i][j]));
            //omp_unset_lock(&lock);
		}
		if (iam<limit)
		{
			for ( ; isync[iam]==1; )
			{
				#pragma omp flush(isync)
			}
			isync[iam]=1;
			#pragma omp flush(isync)
		}
	} 
#pragma omp critical 
    if(maxii > eps) maxii_shared = maxii;
    omp_destroy_lock(&lock);
}
    eps = maxii_shared;
}


void verify()
{ 
	double s;
	s=0.;
	for(i=0; i<=N-1; i++)
	#pragma omp parallel for private(j), shared(A,i), reduction(+:s)
	for(j=0; j<=N-1; j++)
	{
		s=s+A[i][j]*(i+1)*(j+1)/(N*N);
	}
	printf("  S = %f\n",s);
}


