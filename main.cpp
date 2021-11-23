
#include "cmic.h"
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cstdio>
#include <string>
using namespace std;
using namespace libzpaq;

// Handle errors in libzpaq and elsewhere
void libzpaq::error(const char* msg) {
    if (strstr(msg, "ut of memory")) throw std::bad_alloc();
    throw std::runtime_error(msg);
}

void c_test(int argc, char* argv[]) {
    puts("Start Compression...");
    cmic::compressor co(16);
    cmic::param par;
    if(argc >= 6) par.set_threshold(stod(argv[5]));
    if(argc >= 5) par.set_k(stol(argv[4]));
    par.set_outname(argv[3]);
    co.init(par);
    par.set_inname(argv[2]);
    co.qs_compress();
    co.end();
    puts("Compression Completed.");
}

void d_test(int argc, char* argv[]) {
    puts("Start Decompression...");
    cmic::decompressor de(0);
    de.open(argv[2]);
    de.read_format();
    de.read_table();
    de.read_content();
    de.set_out(argv[3]);
    de.get_qs();
    puts("Decompression Completed!");
}

void r_test(int argc, char* argv[]) {
    puts("Start Random decompression...");
    cmic::decompressor de(0);
    de.open(argv[2]);
    de.read_format();
    de.read_table();
    uint32_t l = stoul(argv[4]), r = stoul(argv[5]);
    de.set_out(argv[3]);
    de.query(l, r);
    puts("Random Decompression Completed!");
}

int main(int argc, char* argv[])
{
    if(argc == 1){
	   puts("Try to type 'cmic h' for help.");
    }
    else if(argv[1][0] == 'c') c_test(argc, argv);
    else if(argv[1][0] == 'd') d_test(argc, argv);
    else if(argv[1][0] == 'r') r_test(argc, argv);
    else if(argv[1][0] == 'h') {
        puts("");
        puts("");
        puts("For compression: cmic c <input-file> <output-file>. \nExample: cmic c sample.in sample.cmic.\n");
        puts("");
        puts("");
    }
    else{ 
        puts("Invalid option. Try to type 'cmic h' for help.:");
    	return -1;
    }
    return 0;
}
