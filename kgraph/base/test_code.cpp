#include <cstdlib>
#include <pthread.h>

// Setting
#define INT long
#define REAL float
#include <cstring>
#include <cstdio>
#include <string>

/* in C++ code
INT: long, --> np.int64 --> torch.long
REAL: float, --> np.float --> torch.float
*/


std::string inPath = "../data/FB15K/";
std::string outPath = "../data/FB15K/";
std::string ent_file = "";
std::string rel_file = "";
std::string train_file = "";
std::string valid_file = "";
std::string test_file = "";

// set the dataset dir 
extern "C"
void setInPath(char *path) {
	INT len = strlen(path);
	inPath = "";
	for (INT i = 0; i < len; i++)
		inPath = inPath + path[i];
	printf("Input Files Path : %s\n", inPath.c_str());
}

// set the out dataset dir
extern "C"
void setOutPath(char *path) {
	INT len = strlen(path);
	outPath = "";
	for (INT i = 0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output Files Path : %s\n", outPath.c_str());
}

// set the train dataset path
extern "C"
void setTrainPath(char *path) {
	INT len = strlen(path);
	train_file = "";
	for (INT i = 0; i < len; i++)
		train_file = train_file + path[i];
	printf("Training Files Path : %s\n", train_file.c_str());
}

extern "C"
void setValidPath(char *path) {
	INT len = strlen(path);
	valid_file = "";
	for (INT i = 0; i < len; i++)
		valid_file = valid_file + path[i];
	printf("Valid Files Path : %s\n", valid_file.c_str());
}

extern "C"
void setTestPath(char *path) {
	INT len = strlen(path);
	test_file = "";
	for (INT i = 0; i < len; i++)
		test_file = test_file + path[i];
	printf("Test Files Path : %s\n", test_file.c_str());
}

extern "C"
void setEntPath(char *path) {
	INT len = strlen(path);
	ent_file = "";
	for (INT i = 0; i < len; i++)
		ent_file = ent_file + path[i];
	printf("Entity Files Path : %s\n", ent_file.c_str());
}

extern "C"
void setRelPath(char *path) {
	INT len = strlen(path);
	rel_file = "";
	for (INT i = 0; i < len; i++)
		rel_file = rel_file + path[i];
	printf("Relation Files Path : %s\n", rel_file.c_str());
}

/*
============================================================
*/

// set the number of work threads
INT workThreads = 1;

extern "C"
void setWorkThreads(INT threads) {
	workThreads = threads;
}

// get the number of work threads
extern "C"
INT getWorkThreads() {
	return workThreads;
}

/*
============================================================
*/

INT relationTotal = 0;
INT entityTotal = 0;
INT tripleTotal = 0;
INT testTotal = 0;
INT trainTotal = 0;
INT validTotal = 0;

extern "C"
INT getEntityTotal() {
	return entityTotal;
}

extern "C"
INT getRelationTotal() {
	return relationTotal;
}

extern "C"
INT getTripleTotal() {
	return tripleTotal;
}

extern "C"
INT getTrainTotal() {
	return trainTotal;
}

extern "C"
INT getTestTotal() {
	return testTotal;
}

extern "C"
INT getValidTotal() {
	return validTotal;
}
/*
============================================================
*/

INT bernFlag = 0;

extern "C"
void setBern(INT con) {
	bernFlag = con;
}


// Random
#include <cstdlib>

// the random seeds for all threads.
unsigned long long *next_random;

// reset the random seeds for all threads
extern "C"
void randReset() {
	next_random = (unsigned long long *)calloc(workThreads, sizeof(unsigned long long));
	for (INT i = 0; i < workThreads; i++)
		next_random[i] = rand();
}

// get a random interger for the id-th thread with the corresponding random seed.
unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)(25214903917) + 11;
	return next_random[id];
}

// get a random interger from the range [0,x) for the id-th thread.
INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

// get a random interger from the range [a,b) for the id-th thread.
INT rand(INT a, INT b){
	return (rand() % (b-a))+ a;
}

// Triple
// the Triple class
struct Triple {

	INT h, r, t;

	static bool cmp_head(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.r < b.r)||(a.h == b.h && a.r == b.r && a.t < b.t);
	}

	static bool cmp_tail(const Triple &a, const Triple &b) {
		return (a.t < b.t)||(a.t == b.t && a.r < b.r)||(a.t == b.t && a.r == b.r && a.h < b.h);
	}

	static bool cmp_rel(const Triple &a, const Triple &b) {
		return (a.h < b.h)||(a.h == b.h && a.t < b.t)||(a.h == b.h && a.t == b.t && a.r < b.r);
	}

	static bool cmp_rel2(const Triple &a, const Triple &b) {
		return (a.r < b.r)||(a.r == b.r && a.h < b.h)||(a.r == b.r && a.h == b.h && a.t < b.t);
	}

};




// Reader
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>

INT *freqRel, *freqEnt;
INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;
REAL *left_mean, *right_mean;
REAL *prob;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importProb(REAL temp){
    if (prob != NULL)
        free(prob);
    FILE *fin;
    fin = fopen((inPath + "kl_prob.txt").c_str(), "r");
    printf("Current temperature:%f\n", temp);
    prob = (REAL *)calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i){
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal-1; ++j){
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal-1; ++j){
            prob[i*(relationTotal-1)+j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

extern "C"
void importTrainFiles() {

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &relationTotal);
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

    if (train_file == "")
        fin = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        fin = fopen(train_file.c_str(), "r");
	tmp = fscanf(fin, "%ld", &trainTotal);
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].r);          // t --> r
		tmp = fscanf(fin, "%ld", &trainList[i].t);          // r --> t
	}
	fclose(fin);
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
	for (INT i = 1; i < tmp; i++)
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}

	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);
	printf("The total of train triples is %ld.\n", trainTotal);

	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < trainTotal; i++) {
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

Triple *testList;
Triple *validList;
Triple *tripleList;

extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;
    
    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    FILE* f_kb1, * f_kb2, * f_kb3;
    if (train_file == "")
        f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        f_kb2 = fopen(train_file.c_str(), "r");
    if (test_file == "")
        f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
    else
        f_kb1 = fopen(test_file.c_str(), "r");
    if (valid_file == "")
        f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    else
        f_kb3 = fopen(valid_file.c_str(), "r");
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %ld.\n", testTotal);
    printf("The total of valid triples is %ld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
    testRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < testTotal; i++) {
	if (testList[i].r != testList[i-1].r) {
	    testRig[testList[i-1].r] = i - 1;
	    testLef[testList[i].r] = i;
	}
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *)calloc(relationTotal, sizeof(INT));
    validRig = (INT *)calloc(relationTotal, sizeof(INT));
    memset(validLef, -1, sizeof(INT)*relationTotal);
    memset(validRig, -1, sizeof(INT)*relationTotal);
    for (INT i = 1; i < validTotal; i++) {
	if (validList[i].r != validList[i-1].r) {
	    validRig[validList[i-1].r] = i - 1;
	    validLef[validList[i].r] = i;
	}
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}

INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;
INT* head_type;
INT* tail_type;

extern "C"
void importTypeFiles() {

    head_lef = (INT *)calloc(relationTotal, sizeof(INT));
    head_rig = (INT *)calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
    INT total_lef = 0;
    INT total_rig = 0;
    FILE* f_type = fopen((inPath + "constrain.txt").c_str(),"r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    head_type = (INT *)calloc(total_lef, sizeof(INT)); 
    tail_type = (INT *)calloc(total_rig, sizeof(INT));
    total_lef = 0;
    total_rig = 0;
    f_type = fopen((inPath + "constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif


// Corrupt
#include "Random.h"
#include "Triple.h"
#include "Reader.h"

INT corrupt_head(INT id, INT h, INT r, bool filter_flag = true) {
	/* corrupt the head entity of a triple.
	Parameters:
		id: the thread id
		h: the head entity id
		r: the relation id
	*/
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, entityTotal - 1);
		if (tmp < h)
			return tmp;
		else
			return tmp + 1;
	}
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainHead[ll].t) return tmp;
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}

INT corrupt_tail(INT id, INT t, INT r, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, entityTotal - 1);
		if (tmp < t)
			return tmp;
		else
			return tmp + 1;
	}
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp = rand_max(id, entityTotal - (rr - ll + 1));
	if (tmp < trainTail[ll].h) return tmp;
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


INT corrupt_rel(INT id, INT h, INT t, INT r, bool p = false, bool filter_flag = true) {
	INT lef, rig, mid, ll, rr;
	if (not filter_flag) {
		INT tmp = rand_max(id, relationTotal - 1);
		if (tmp < r)
			return tmp;
		else
			return tmp + 1;
	}
	lef = lefRel[h] - 1;
	rig = rigRel[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;
	lef = lefRel[h];
	rig = rigRel[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;
	INT tmp;
	if(p == false) {	
		tmp = rand_max(id, relationTotal - (rr - ll + 1));
	}
	else {
		INT start = r * (relationTotal - 1);
		REAL sum = 1;
		bool *record = (bool *)calloc(relationTotal - 1, sizeof(bool));
		for (INT i = ll; i <= rr; ++i){
			if (trainRel[i].r > r){
				sum -= prob[start + trainRel[i].r-1];
				record[trainRel[i].r-1] = true;
			}
			else if (trainRel[i].r < r){
				sum -= prob[start + trainRel[i].r];
				record[trainRel[i].r] = true;
			}
		}		
		REAL *prob_tmp = (REAL *)calloc(relationTotal-(rr-ll+1), sizeof(REAL));
		INT cnt = 0;
		REAL rec = 0;
		for (INT i = start; i < start + relationTotal - 1; ++i) {
			if (record[i-start])
				continue;
			rec += prob[i] / sum;
			prob_tmp[cnt++] = rec;
		}
		REAL m = rand_max(id, 10000) / 10000.0;
		lef = 0;
		rig = cnt - 1;
		while (lef < rig) {
			mid = (lef + rig) >> 1;
			if (prob_tmp[mid] < m) 
				lef = mid + 1; 
			else
				rig = mid;
		}
		tmp = rig;
		free(prob_tmp);
		free(record);
	}
	if (tmp < trainRel[ll].r) return tmp;
	if (tmp > trainRel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		if (trainRel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	return tmp + lef - ll + 1;
}


bool _find(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = tripleTotal - 1;
    INT mid;
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
    return false;
}

INT corrupt(INT h, INT r){
	INT ll = tail_lef[r];
	INT rr = tail_rig[r];
	INT loop = 0;
	INT t;
	while(true) {
		t = tail_type[rand(ll, rr)];
		if (not _find(h, t, r)) {
			return t;
		} else {
			loop ++;
			if (loop >= 1000) {
				return corrupt_head(0, h, r);
			}
		} 
	}
}



// Test
#ifndef TEST_H
#define TEST_H

/*=====================================================================================
link prediction
======================================================================================*/
INT lastHead = 0;
INT lastTail = 0;
INT lastRel = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

REAL l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
REAL l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
REAL hit1, hit3, hit10, mr, mrr;
REAL hit1TC, hit3TC, hit10TC, mrTC, mrrTC;

extern "C"
void initTest() {
    lastHead = 0;
    lastTail = 0;
    lastRel = 0;
    l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l_tot = 0, r_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
    l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l_filter_tot = 0, r_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
    REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

    l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l_tot_constrain = 0, r_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
    l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l_filter_tot_constrain = 0, r_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
}

extern "C"
void getHeadBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = i;
        pt[i] = testList[lastHead].t;
        pr[i] = testList[lastHead].r;
    }
    lastHead++;
}

extern "C"
void getTailBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < entityTotal; i++) {
        ph[i] = testList[lastTail].h;
        pt[i] = i;
        pr[i] = testList[lastTail].r;
    }
    lastTail++;
}

extern "C"
void getRelBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < relationTotal; i++) {
        ph[i] = testList[lastRel].h;
        pt[i] = testList[lastRel].t;
        pr[i] = i;
    }
}

extern "C"
void testHead(REAL *con, INT lastHead, bool type_constrain = false) {
    INT h = testList[lastHead].h;
    INT t = testList[lastHead].t;
    INT r = testList[lastHead].r;
    INT lef, rig;
    if (type_constrain) {
        lef = head_lef[r];
        rig = head_rig[r];
    }
    REAL minimal = con[h];
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    for (INT j = 0; j < entityTotal; j++) {
        if (j != h) {
            REAL value = con[j];
            if (value < minimal) {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
            if (type_constrain) {
                while (lef < rig && head_type[lef] < j) lef ++;
                if (lef < rig && j == head_type[lef]) {
                    if (value < minimal) {
                        l_s_constrain += 1;
                        if (not _find(j, t, r)) {
                            l_filter_s_constrain += 1;
                        }
                    }  
                }
            }
        }
    }

    if (l_filter_s < 10) l_filter_tot += 1;
    if (l_s < 10) l_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 1) l1_tot += 1;

    l_filter_rank += (l_filter_s+1);
    l_rank += (1 + l_s);
    l_filter_reci_rank += 1.0/(l_filter_s+1);
    l_reci_rank += 1.0/(l_s+1);

    if (type_constrain) {
        if (l_filter_s_constrain < 10) l_filter_tot_constrain += 1;
        if (l_s_constrain < 10) l_tot_constrain += 1;
        if (l_filter_s_constrain < 3) l3_filter_tot_constrain += 1;
        if (l_s_constrain < 3) l3_tot_constrain += 1;
        if (l_filter_s_constrain < 1) l1_filter_tot_constrain += 1;
        if (l_s_constrain < 1) l1_tot_constrain += 1;

        l_filter_rank_constrain += (l_filter_s_constrain+1);
        l_rank_constrain += (1+l_s_constrain);
        l_filter_reci_rank_constrain += 1.0/(l_filter_s_constrain+1);
        l_reci_rank_constrain += 1.0/(l_s_constrain+1);
    }
}

extern "C"
void testTail(REAL *con, INT lastTail, bool type_constrain = false) {
    INT h = testList[lastTail].h;
    INT t = testList[lastTail].t;
    INT r = testList[lastTail].r;
    INT lef, rig;
    if (type_constrain) {
        lef = tail_lef[r];
        rig = tail_rig[r];
    }
    REAL minimal = con[t];
    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;
    for (INT j = 0; j < entityTotal; j++) {
        if (j != t) {
            REAL value = con[j];
            if (value < minimal) {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
            if (type_constrain) {
                while (lef < rig && tail_type[lef] < j) lef ++;
                if (lef < rig && j == tail_type[lef]) {
                        if (value < minimal) {
                            r_s_constrain += 1;
                            if (not _find(h, j ,r)) {
                                r_filter_s_constrain += 1;
                            }
                        }
                }
            }
        }
        
    }

    if (r_filter_s < 10) r_filter_tot += 1;
    if (r_s < 10) r_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 1) r1_tot += 1;

    r_filter_rank += (1+r_filter_s);
    r_rank += (1+r_s);
    r_filter_reci_rank += 1.0/(1+r_filter_s);
    r_reci_rank += 1.0/(1+r_s);
    
    if (type_constrain) {
        if (r_filter_s_constrain < 10) r_filter_tot_constrain += 1;
        if (r_s_constrain < 10) r_tot_constrain += 1;
        if (r_filter_s_constrain < 3) r3_filter_tot_constrain += 1;
        if (r_s_constrain < 3) r3_tot_constrain += 1;
        if (r_filter_s_constrain < 1) r1_filter_tot_constrain += 1;
        if (r_s_constrain < 1) r1_tot_constrain += 1;

        r_filter_rank_constrain += (1+r_filter_s_constrain);
        r_rank_constrain += (1+r_s_constrain);
        r_filter_reci_rank_constrain += 1.0/(1+r_filter_s_constrain);
        r_reci_rank_constrain += 1.0/(1+r_s_constrain);
    }
}

extern "C"
void testRel(REAL *con) {
    INT h = testList[lastRel].h;
    INT t = testList[lastRel].t;
    INT r = testList[lastRel].r;

    REAL minimal = con[r];
    INT rel_s = 0;
    INT rel_filter_s = 0;

    for (INT j = 0; j < relationTotal; j++) {
        if (j != r) {
            REAL value = con[j];
            if (value < minimal) {
                rel_s += 1;
                if (not _find(h, t, j))
                    rel_filter_s += 1;
            }
        }
    }

    if (rel_filter_s < 10) rel_filter_tot += 1;
    if (rel_s < 10) rel_tot += 1;
    if (rel_filter_s < 3) rel3_filter_tot += 1;
    if (rel_s < 3) rel3_tot += 1;
    if (rel_filter_s < 1) rel1_filter_tot += 1;
    if (rel_s < 1) rel1_tot += 1;

    rel_filter_rank += (rel_filter_s+1);
    rel_rank += (1+rel_s);
    rel_filter_reci_rank += 1.0/(rel_filter_s+1);
    rel_reci_rank += 1.0/(rel_s+1);

    lastRel++;
}


extern "C"
void test_link_prediction(bool type_constrain = false) {
    l_rank /= testTotal;
    r_rank /= testTotal;
    l_reci_rank /= testTotal;
    r_reci_rank /= testTotal;
 
    l_tot /= testTotal;
    l3_tot /= testTotal;
    l1_tot /= testTotal;
 
    r_tot /= testTotal;
    r3_tot /= testTotal;
    r1_tot /= testTotal;

    // with filter
    l_filter_rank /= testTotal;
    r_filter_rank /= testTotal;
    l_filter_reci_rank /= testTotal;
    r_filter_reci_rank /= testTotal;
 
    l_filter_tot /= testTotal;
    l3_filter_tot /= testTotal;
    l1_filter_tot /= testTotal;
 
    r_filter_tot /= testTotal;
    r3_filter_tot /= testTotal;
    r1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l_tot, l3_tot, l1_tot);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r_tot, r3_tot, r1_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l_tot+r_tot)/2, (l3_tot+r3_tot)/2, (l1_tot+r1_tot)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l_filter_tot, l3_filter_tot, l1_filter_tot);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r_filter_tot, r3_filter_tot, r1_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l_filter_tot+r_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l1_filter_tot+r1_filter_tot)/2);

    mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
    mr = (l_filter_rank+r_filter_rank) / 2;
    hit10 = (l_filter_tot+r_filter_tot) / 2;
    hit3 = (l3_filter_tot+r3_filter_tot) / 2;
    hit1 = (l1_filter_tot+r1_filter_tot) / 2;

    if (type_constrain) {
        //type constrain
        l_rank_constrain /= testTotal;
        r_rank_constrain /= testTotal;
        l_reci_rank_constrain /= testTotal;
        r_reci_rank_constrain /= testTotal;
     
        l_tot_constrain /= testTotal;
        l3_tot_constrain /= testTotal;
        l1_tot_constrain /= testTotal;
     
        r_tot_constrain /= testTotal;
        r3_tot_constrain /= testTotal;
        r1_tot_constrain /= testTotal;

        // with filter
        l_filter_rank_constrain /= testTotal;
        r_filter_rank_constrain /= testTotal;
        l_filter_reci_rank_constrain /= testTotal;
        r_filter_reci_rank_constrain /= testTotal;
     
        l_filter_tot_constrain /= testTotal;
        l3_filter_tot_constrain /= testTotal;
        l1_filter_tot_constrain /= testTotal;
     
        r_filter_tot_constrain /= testTotal;
        r3_filter_tot_constrain /= testTotal;
        r1_filter_tot_constrain /= testTotal;

        printf("type constraint results:\n");
        
        printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
        printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank_constrain, l_rank_constrain, l_tot_constrain, l3_tot_constrain, l1_tot_constrain);
        printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank_constrain, r_rank_constrain, r_tot_constrain, r3_tot_constrain, r1_tot_constrain);
        printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
                (l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l_tot_constrain+r_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2);
        printf("\n");
        printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank_constrain, l_filter_rank_constrain, l_filter_tot_constrain, l3_filter_tot_constrain, l1_filter_tot_constrain);
        printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank_constrain, r_filter_rank_constrain, r_filter_tot_constrain, r3_filter_tot_constrain, r1_filter_tot_constrain);
        printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
                (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l_filter_tot_constrain+r_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2);

        mrrTC = (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2;
        mrTC = (l_filter_rank_constrain+r_filter_rank_constrain) / 2;
        hit10TC = (l_filter_tot_constrain+r_filter_tot_constrain) / 2;
        hit3TC = (l3_filter_tot_constrain+r3_filter_tot_constrain) / 2;
        hit1TC = (l1_filter_tot_constrain+r1_filter_tot_constrain) / 2;
    }
}

extern "C"
void test_relation_prediction() {
    rel_rank /= testTotal;
    rel_reci_rank /= testTotal;
  
    rel_tot /= testTotal;
    rel3_tot /= testTotal;
    rel1_tot /= testTotal;

    // with filter
    rel_filter_rank /= testTotal;
    rel_filter_reci_rank /= testTotal;
  
    rel_filter_tot /= testTotal;
    rel3_filter_tot /= testTotal;
    rel1_filter_tot /= testTotal;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            rel_reci_rank, rel_rank, rel_tot, rel3_tot, rel1_tot);
    printf("\n");
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            rel_filter_reci_rank, rel_filter_rank, rel_filter_tot, rel3_filter_tot, rel1_filter_tot);
}

extern "C"
REAL getTestLinkHit10(bool type_constrain = false) {
    if (type_constrain)
        return hit10TC;
    printf("%f\n", hit10);
    return hit10;
}

extern "C"
REAL  getTestLinkHit3(bool type_constrain = false) {
    if (type_constrain)
        return hit3TC;
    return hit3;
}

extern "C"
REAL  getTestLinkHit1(bool type_constrain = false) {
    if (type_constrain)
        return hit1TC;    
    return hit1;
}

extern "C"
REAL  getTestLinkMR(bool type_constrain = false) {
    if (type_constrain)
        return mrTC;
    return mr;
}

extern "C"
REAL  getTestLinkMRR(bool type_constrain = false) {
    if (type_constrain)
        return mrrTC;    
    return mrr;
}


/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList = NULL;

extern "C"
void getNegTest() {
    if (negTestList == NULL)
        negTestList = (Triple *)calloc(testTotal, sizeof(Triple));
    for (INT i = 0; i < testTotal; i++) {
        negTestList[i] = testList[i];
        if (randd(0) % 1000 < 500)
            negTestList[i].t = corrupt_head(0, testList[i].h, testList[i].r);
        else
            negTestList[i].h = corrupt_tail(0, testList[i].t, testList[i].r);
    }
}

extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < testTotal; i++) {
        ph[i] = testList[i].h;
        pt[i] = testList[i].t;
        pr[i] = testList[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}
#endif









extern "C"
void setInPath(char *path);

extern "C"
void setTrainPath(char *path);

extern "C"
void setValidPath(char *path);

extern "C"
void setTestPath(char *path);

extern "C"
void setEntPath(char *path);

extern "C"
void setRelPath(char *path);

extern "C"
void setOutPath(char *path);

extern "C"
void setWorkThreads(INT threads);

extern "C"
void setBern(INT con);

extern "C"
INT getWorkThreads();

extern "C"
INT getEntityTotal();

extern "C"
INT getRelationTotal();

extern "C"
INT getTripleTotal();

extern "C"
INT getTrainTotal();

extern "C"
INT getTestTotal();

extern "C"
INT getValidTotal();

extern "C"
void randReset();

extern "C"
void importTrainFiles();

struct Parameter {
	INT id;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	REAL *batch_y;
	INT batchSize;
	INT negRate;
	INT negRelRate;
	bool p;
	bool val_loss;
	INT mode;
	bool filter_flag;
};

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
	INT lef, rig;
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	if (val_loss == false) {
		for (INT batch = lef; batch < rig; batch++) {
			INT i = rand_max(id, trainTotal);
			batch_h[batch] = trainList[i].h;
			batch_t[batch] = trainList[i].t;
			batch_r[batch] = trainList[i].r;
			batch_y[batch] = 1;
			INT last = batchSize;
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					if (bernFlag)
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) {
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				} else {
					if(mode == -1){
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				}
			}
			for (INT times = 0; times < negRelRate; times++) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
				batch_y[batch + last] = -1;
				last += batchSize;
			}
		}
	}
	else
	{
		for (INT batch = lef; batch < rig; batch++)
		{
			batch_h[batch] = validList[batch].h;
			batch_t[batch] = validList[batch].t;
			batch_r[batch] = validList[batch].r;
			batch_y[batch] = 1;
		}
	}
	pthread_exit(NULL);
}

extern "C"
void sampling(
		INT *batch_h, 
		INT *batch_t, 
		INT *batch_r, 
		REAL *batch_y, 
		INT batchSize, 
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);

	free(pt);
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}