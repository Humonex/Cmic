
#include "cmic.h"
#include "libzpaq.h"
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <pthread.h>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <unordered_map>
#include <queue>
#include <malloc.h>

using namespace libzpaq;
using namespace std;

#define MAXTABLESIZE 10000 
namespace cmic {

typedef void* ThreadReturn;                               
void run(ThreadID& tid, ThreadReturn(*f)(void*), void* arg)
    {pthread_create(&tid, NULL, f, arg);}
void join(ThreadID tid) {pthread_join(tid, NULL);}        
typedef pthread_mutex_t Mutex;                             
void init_mutex(Mutex& m) {pthread_mutex_init(&m, 0);}     
void lock(Mutex& m) {pthread_mutex_lock(&m);}             
void release(Mutex& m) {pthread_mutex_unlock(&m);}        
void destroy_mutex(Mutex& m) {pthread_mutex_destroy(&m);}  

typedef int ElementType;
struct LNode
{
	ElementType data;
	LNode *next;
};
typedef LNode *PtrToNode;
typedef PtrToNode LinkList;
struct TblNode
{
	int tablesize;  
	LinkList heads; 
};
typedef struct TblNode *HashTable;

int NextPrime(int n)
{
	int p = (n % 2) ? n + 2 : n + 1;
	int i;
	while (p <= MAXTABLESIZE)
	{
		for (i = (int)sqrt(p); i > 2; i--)
		{
			if ((p % i) == 0)
				break;
		}
		if (i == 2)
			break; 
		else
			p += 2;
	}
	return p;
}


HashTable CreateTable(int table_size)
{
	HashTable h = (HashTable)malloc(sizeof(TblNode));
	h->tablesize = NextPrime(table_size);
	h->heads = (LinkList)malloc(h->tablesize * sizeof(LNode));
	for (int i = 0; i < h->tablesize; i++)
	{
		h->heads[i].next = NULL;
	}
	return h;
}

int Hash(ElementType key, int n)
{
	return key % 11;
}

LinkList Find(HashTable h, ElementType key1, ElementType key2)
{
	int pos1;
	int pos2;
	pos1 = Hash(key1, h->tablesize);
	pos2 = Hash(key2, h->tablesize);
	LinkList p = h->heads[pos1].next;
	uint32_t l = 0xffffffffu, r = 0;
	for (int m = 0; m < h->tablesize; m++) {
		LinkList p = h->heads[m].next;
		while (p)
		{
			if (i == b.bucket && p->data > L && p->data < R) {
				in[j] = true;
				l = min(l, b.eline);
				r = max(r, b.eline);
			}
			p = p->next;
		}
	}
	auto &_val = job->val[i];
	job->mx[i] = lower_bound(_val.begin(), _val.end(), L + 1) - _val.begin() - 1;
	job->flag[i][job->mx[i]] = 2;
	L = l;
	l = r; r = fmt.qlen[(i + 1) % 2];
	for (auto& b : job->binfo) if (i == b.bucket) {
		if (b.eline > l && b.eline < r) r = b.eline;
	}
	R = r;
	return R;
}


bool Insert(HashTable h, ElementType key)
{
	LinkList p = Find(h, key); 
	if (!p)
	{
		LinkList new_cell = (LinkList)malloc(sizeof(LNode));
		new_cell->data = key;
		int pos = Hash(key, h->tablesize);
		new_cell->next = h->heads[pos].next;
		h->heads[pos].next = new_cell;
		return true;
	}
	else
	{
		cout << "This address already exists " << endl;
		return false;
	}
}

/*Destroy this table*/
void DestroyTable(HashTable h)
{
	int i;
	LinkList p, tmp;
	//释放每个节点
	for (i = 0; i < h->tablesize; i++)
	{
		p = h->heads[i].next;
		while (p)
		{
			tmp = p->next;
			free(p);
			p = tmp;
		}
	}
	free(h->heads);
	free(h);
}
class Semaphore {
public:
    Semaphore() {sem=-1;}
    void init(int n) {
        assert(n>=0);
        assert(sem==-1);
        pthread_cond_init(&cv, 0);
        pthread_mutex_init(&mutex, 0);
        sem=n;
    }
    void destroy() {
        assert(sem>=0);
        pthread_mutex_destroy(&mutex);
        pthread_cond_destroy(&cv);
    }
    int wait() {
        assert(sem>=0);
        pthread_mutex_lock(&mutex);
        int r=0;
        if (sem==0) r=pthread_cond_wait(&cv, &mutex);
        assert(sem>0);
        --sem;
        pthread_mutex_unlock(&mutex);
        return r;
    }
    void signal() {
        assert(sem>=0);
        pthread_mutex_lock(&mutex);
        ++sem;
        pthread_cond_signal(&cv);
        pthread_mutex_unlock(&mutex);
    }
private:
    pthread_cond_t cv;  // to signal FINISHED
    pthread_mutex_t mutex; // protects cv
    int sem;  // semaphore count
};

struct BlockInfo {
    enum Tag {BASE} tag;
    int64_t pos;
    uint32_t length;
    uint32_t start;
    uint32_t end;
    uint32_t eline;
    uint8_t bucket;

    void write(FILE* f) {
        char _tag = tag;
        fwrite(&_tag, sizeof _tag, 1, f);
        fwrite(&pos, sizeof pos, 1, f);
        fwrite(&length, sizeof length, 1, f);
        fwrite(&start, sizeof start, 1, f);
        fwrite(&end, sizeof end, 1, f);
        fwrite(&eline, sizeof eline, 1, f);
        fwrite(&bucket, sizeof bucket, 1, f);
    }

    bool read(FILE* f) {
        char _tag;
        if(fread(&_tag, sizeof _tag, 1, f) == 0) return false;
        tag = Tag(_tag);
        fread(&pos, sizeof pos, 1, f);
        fread(&length, sizeof length, 1, f);
        fread(&start, sizeof start, 1, f);
        fread(&end, sizeof end, 1, f);
        fread(&eline, sizeof eline, 1, f);
        fread(&bucket, sizeof bucket, 1, f);
        return true;
    }
};
struct IndexInfo: public BlockInfo {
	enum Tag { BASE } tag;
	int64_t pos;
	uint32_t length;
	uint32_t start;
	uint32_t eline;
	uint8_t bucket;
	int64_t n;
	void write(FILE* f) {}
	bool read(FILE* f) {}
};
// A CompressJob is a queue of blocks to compress and write to the archive.
// Each block cycles through states EMPTY, FILLING, FULL, COMPRESSING,
// COMPRESSED, WRITING. The main thread waits for EMPTY buffers and
// fills them. A set of compressThreads waits for FULL threads and compresses
// them. A writeThread waits for COMPRESSED buffers at the front
// of the queue and writes and removes them.

// Buffer queue element
struct CJ {
    BlockInfo info;
	IndexInfo info2;
    enum {EMPTY, FULL, COMPRESSING, COMPRESSED, WRITING} state;
    StringBuffer in;       // uncompressed input
    StringBuffer out;      // compressed output
    string comment;        // if "" use default
    string method;         // compression level or "" to mark end of data
    Semaphore full;        // 1 if in is FULL of data ready to compress
    CJ(): state(EMPTY) {}
};

struct FileWriter: public libzpaq::Writer {
    FILE* fp;
    FileWriter(const char* filename) {
        fp = fopen(filename, "wb");
    }

    ~FileWriter() {
        if(fp) {
            fclose(fp);
            fp = NULL;
        }
    }

    void seek(int offset) {
        fseek(fp, offset, SEEK_SET);
    }

    void put(int c) {
        putc(c, fp);
    }

    void write(const char* buf, int n) {
        fwrite(buf, 1, n, fp);
    }

    int64_t tell() {
        return ftello(fp);
    }
};

// Instructions to a compression job
class CompressJob {
public:
    Mutex mutex;           
    FileWriter* out;
    char score;
private:
    int job;              
    CJ* q;                 
    unsigned qsize;       
    int front;            
    Semaphore empty;       
    Semaphore compressors; 
    Semaphore compressed;
    queue<int> wq;
    bool finished;
    int count;
public:
    friend ThreadReturn compressThread(void* arg);
    friend ThreadReturn writeThread(void* arg);
    CompressJob(int threads, int buffers): job(0), q(0), qsize(buffers), front(0), finished(false), count(0) {
        q=new CJ[buffers];
        if (!q) throw std::bad_alloc();
        init_mutex(mutex);
        empty.init(buffers);
        compressors.init(threads);
        compressed.init(0);
        for (int i=0; i<buffers; ++i) {
            q[i].full.init(0);
        }
    }
    ~CompressJob() {
        for (int i=qsize-1; i>=0; --i) {
            q[i].full.destroy();
        }
        compressed.destroy();
        compressors.destroy();
        empty.destroy();
        destroy_mutex(mutex);
        delete[] q;
    }      
    void write(StringBuffer& s, BlockInfo::Tag _tag, uint8_t bucket, uint32_t start, uint32_t end, uint32_t eline, string method);
	void Index (StringBuffer& s, IndexInfo::Tag _tag, uint8_t bucket, uint32_t start, uint32_t eline, string method);
    vector<BlockInfo> binfo;
    void write_binfo() {
        for(auto& b : binfo) {
            b.write(out->fp);
        }
    }
};

void CompressJob::write(StringBuffer& s, BlockInfo::Tag _tag, uint8_t bucket, uint32_t start, uint32_t end, uint32_t eline, string method) {
    for (unsigned k=(method=="")?qsize:1; k>0; --k) {
        empty.wait();
        lock(mutex);
        ++count;
        unsigned i, j;
        for (i=0; i<qsize; ++i) {
            if (q[j=(i+front)%qsize].state==CJ::EMPTY) {
                q[j].info.tag=_tag;
                q[j].info.bucket=bucket;
                q[j].info.start=start;
                q[j].info.end=end;
                q[j].info.eline=eline;
                q[j].comment="";
                q[j].method=method;
                q[j].in.resize(0);
                q[j].in.swap(s);
                q[j].state=CJ::FULL;
                q[j].full.signal();
                break;
            }
        }
        release(mutex);
        assert(i<qsize);  
    }
}
void CompressJob::Index (StringBuffer& s, IndexInfo::Tag _tag, uint8_t bucket, uint32_t start, uint32_t eline, string method) {
	for (unsigned k = (method == "") ? qsize : 1; k>0; --k) {
		empty.wait();
		lock(mutex);
		++count;
		unsigned i, j;
		for (i = 0; i<qsize; ++i) {
			if (q[j = (i + front) % qsize].state == CJ::EMPTY) {
				q[j].info2.tag = _tag;
				q[j].info2.bucket = bucket;
				q[j].info2.start = start;
				q[j].info2.eline = eline;
				q[j].comment = "";
				q[j].method = method;
				q[j].in.resize(0);
				q[j].in.swap(s);
				q[j].state = CJ::FULL;
				q[j].full.signal();
				if (score == "~") {
					q[j].info2.n++;
				}
				q[j].info2.end = q[j].info2.start + count + q[j].info2.n;
				break;
			}
		}
		release(mutex);
		assert(i<qsize);
	}
}

void pack(StringBuffer& in, char score) {
    StringBuffer out;
    out.resize(in.size() / 2);
    out.resize(0);
	if (score == 'A') {
		score = 0;
	}
	else if (score == 'T') {
		score = 1;
	}
	else if (score == 'C') {
		score = 2;
	}
	else if (score == 'G') {
		score = 3;
	}
	else {
		score = 4;
	}
    out.swap(in);
}

// Compress data in the background, one per buffer
ThreadReturn compressThread(void* arg) {
    CompressJob& job=*(CompressJob*)arg;
    int jobNumber=0;
    try {

        // Get job number = assigned position in queue
        lock(job.mutex);
        jobNumber=job.job++;
        assert(jobNumber>=0 && jobNumber<int(job.qsize));
        CJ& cj=job.q[jobNumber];
        release(job.mutex);

        // Work until done
        while (true) {
            cj.full.wait();
            lock(job.mutex);

            // Check for end of input
            if (cj.method=="") {
                job.wq.push(jobNumber);
                job.compressed.signal();
                release(job.mutex);
                return 0;
            }

            // Compress
            assert(cj.state==CJ::FULL);
            cj.state=CJ::COMPRESSING;
            release(job.mutex);
            job.compressors.wait();
            if(cj.info.tag == BlockInfo::BASE) pack(cj.in, job.score);
            libzpaq::compressBlock(&cj.in, &cj.out, cj.method.c_str(), "", cj.comment.c_str(), false);
            cj.in.resize(0);
            lock(job.mutex);
            cj.state=CJ::COMPRESSED;
            job.wq.push(jobNumber);
            job.compressed.signal();
            job.compressors.signal();
            release(job.mutex);
        }
    }
    catch (std::exception& e) {
        lock(job.mutex);
        fflush(stdout);
        fprintf(stderr, "job %d: %s\n", jobNumber+1, e.what());
        release(job.mutex);
        exit(1);
    }
    return 0;
}

void compressor::init(param _par) {
    par = _par;
    job->out = new FileWriter(par.out_name);
    job->out->seek(17);
}

// Write compressed data in the background
ThreadReturn writeThread(void* arg) {
    CompressJob& job=*(CompressJob*)arg;
    try {

        // work until done
        while (true) {
            if(job.finished && job.count == 0) return 0;

            // wait for something to write
            job.compressed.wait();
            lock(job.mutex);
            CJ& cj=job.q[job.wq.front()];
            job.wq.pop();
            --job.count;

            // Quit if end of input
            if (cj.method=="") {
                job.finished = true;
                release(job.mutex);
                continue;
            }

            // Write
            assert(cj.state==CJ::COMPRESSED);
            cj.state=CJ::WRITING;
            if (job.out && cj.out.size()>0) {
                release(job.mutex);
                assert(cj.out.c_str());
                const char* p=cj.out.c_str();
                uint32_t n=cj.out.size();
                const uint32_t N=1<<30;
                cj.info.pos=job.out->tell();
                while (n>N) {
                    job.out->write(p, N);
                    p+=N;
                    n-=N;
                }
                job.out->write(p, n);
                cj.info.length=job.out->tell() - cj.info.pos;
                lock(job.mutex);
            }
            cj.out.resize(0);
            cj.state=CJ::EMPTY;
            job.front=(job.front+1)%job.qsize;
            job.binfo.push_back(cj.info);
            job.empty.signal();
            release(job.mutex);
        }
    }
    catch (std::exception& e) {
        fflush(stdout);
        fprintf(stderr, "zpaq exiting from writeThread: %s\n", e.what());
        exit(1);
    }
    return 0;
}

const size_t BUFFER_SIZE = 1 << 25;
const char METHOD[] = "55,220,0";

compressor::compressor(int _threads): threads(_threads) {
    if(threads < 1) threads = thread::hardware_concurrency();
    tid.resize(threads*2-1);
    job = new CompressJob(threads, tid.size());
    for (unsigned i=0; i<tid.size(); ++i) run(tid[i], compressThread, job);
    run(wid, writeThread, job);
}

void compressor::end() {
    StringBuffer _;
    job->write(_, BlockInfo::Tag(0), 0, 0, 0, 0, "");  // signal end of input
    for (unsigned i=0; i<tid.size(); ++i) join(tid[i]);
    join(wid);
    int64_t len = job->out->tell();
    job->write_binfo();
    job->out->seek(0);
    fmt.write(job->out->fp);
    fwrite(&len, sizeof len, 1, job->out->fp);
    delete job->out;
    delete job;
}

void compressor::get_score(const vector<string>& sample) {
    int score_cnt[128];
    memset(score_cnt, 0, sizeof score_cnt);
    for(auto& s : sample) {
        for(char c : s) ++score_cnt[c];
    }
    fmt.score = max_element(score_cnt, score_cnt + 128) - score_cnt;
}

double compressor::get_table(const vector<string>& sample, unordered_map<long long, double>& table, int k) {
    unordered_map<long long, int> mp;
    int tot = 0;
    for(auto& s : sample) {
        for(size_t i = k-1; i < s.size(); ++i) {
            long long val = 0;
            for(size_t l = i+1-k; l <= i; ++l) val = val << 7 | s[l];
            ++mp[val];
		}
		tot += s.size()+1-k;
    }
    vector<pair<int, long long>> vec;
    for(auto& _ : mp) vec.emplace_back(_.second, _.first);
    sort(vec.begin(), vec.end(), greater<pair<int, long long>>());
    int cnt = tot * 0.7;
    for(auto it = vec.begin(); cnt > 0; ++it) {
        table[it->second] = it->first / (double)tot;
        cnt -= it->first;
    }
    double mx = 0;
    for(auto& s : sample) {
        double score = 0;
        for(size_t i = k-1; i < s.size(); ++i) {
            long long val = 0;
            for(size_t l = i+1-k; l <= i; ++l) val = val << 7 | s[l];
            score += table[val];
        }
        mx = max(mx, score/(s.size()+1-k));
    }
    return mx;
}

char _s[51234];

void compressor::qs_compress() {
    double border;
    const double threshold = par.threshold;
    const int k = par.k;
    unordered_map<long long, double> table;
    table.max_load_factor(0.5);
    uint32_t num = 0;
    string s;
    vector<string> sample(100000, s);
    {
        while(num < 100000 && gets(_s)) sample[num++] = _s;
        sample.resize(num);
        get_score(sample);
        border = get_table(sample, table, k) * par.threshold;
        job->score = fmt.score;
    }
    StringBuffer sb[2];
    uint32_t cur[3]{}, pre[2]{}, eline[2]{};
    long long base = 1;
    for(int i = 1; i < k; ++i) base = base << 7;
    base -= 1;
    int flag = 0;
    num = 0;
    while(true) {
        if(num < sample.size()) s.swap(sample[num++]);
        else if(gets(_s)) s = _s;
        else break;
        long long val = 0;
        for(int i = 0; i < k-1; ++i) val = val << 7 | s[i];
        double score = 0;
        for(size_t j = k-1; j < s.size(); ++j) {
            val = (val & base) << 7 | s[j];
            auto it = table.find(val);
            if(it != table.end()) score += it->second;
        }
        size_t res = score < border*(s.size()+1-k);
        StringBuffer& temp = sb[res];
        ++cur[res];
        temp.write(s.c_str(), s.size());
        temp.put('\n');
        if(res > 0) {
            sb[0].put('\n');
            ++cur[0];
        }
        if(temp.size() > BUFFER_SIZE) {
            job->write(temp, BlockInfo::BASE, res, pre[res], cur[res], eline[res], METHOD);
            pre[res] = cur[res];
            eline[res] = cur[res+1];
            if(res) ++flag;
            else flag = 0;
        }
        if(flag >= 10) {
            job->write(sb[0], BlockInfo::BASE, 0, pre[0], cur[0], eline[0], METHOD);
            pre[0] = cur[0];
            eline[0] = cur[1];
            flag = 0;
        }
    }
    for(size_t i = 0; i < 2; ++i) {
        if(sb[i].size() > 0) {
            job->write(sb[i], BlockInfo::BASE, i, pre[i], cur[i], eline[i], METHOD);
        }
    }
    for(int i = 0; i < 2; ++i) fmt.qlen[i] = cur[i];
}
 
struct Block {
    StringBuffer* in;
    enum {READY, WORKING, GOOD, BAD} state;
    int id, info;
    Block(int _id = -1, int _info = 0): state(READY), id(_id), info(_info), in(new StringBuffer) {}

    void operator = (const Block& b) {
        in = b.in; state = b.state; id = b.id;
    }
};

struct ExtractJob {         // list of jobs
    Mutex mutex;              // protects state
    Mutex read_mutex;
    int job;                  // number of jobs started
    FILE *fp, *fout;
    uint32_t L, R;
    vector<Block> block;      // list of data blocks to extract
    vector<BlockInfo> binfo;
    vector<string>* qout;
    vector<int8_t> flag[2];
    vector<uint32_t> val[2];
    uint32_t cur[2];
    uint32_t mx[2];
    ExtractJob(): job(0) {
        init_mutex(mutex);
        init_mutex(read_mutex);
    }
    ~ExtractJob() {
        destroy_mutex(mutex);
        destroy_mutex(read_mutex);
    }
};
}
