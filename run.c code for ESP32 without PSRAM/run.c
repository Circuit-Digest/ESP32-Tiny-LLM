/* Inference for Llama-2 Transformer model in pure C (Flash Direct-Streaming for Non-PSRAM ESP32) */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>

#ifdef ESP32
    #include "freertos/FreeRTOS.h"
    #include "freertos/task.h"
    #define heap_alloc(size) malloc(size)
#else
    #define heap_alloc(size) malloc(size)
#endif

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim;            // transformer dimension
    int hidden_dim;     // for ffn layers
    int n_layers;       // number of layers
    int n_heads;        // number of query heads
    int n_kv_heads;     // number of key/value heads
    int vocab_size;     // vocabulary size
    int seq_len;        // max sequence length
} Config;

typedef struct {
    size_t token_embedding_table;
    size_t rms_att_weight;
    size_t rms_ffn_weight;
    size_t wq;
    size_t wk;
    size_t wv;
    size_t wo;
    size_t w1;
    size_t w2;
    size_t w3;
    size_t rms_final_weight;
    size_t wcls;
} TransformerWeightsOffsets;

typedef struct {
    float *x;           // activation at current time stamp
    float *xb;          // residual branch activation
    float *xb2;         // convenience buffer
    float *hb;          // hidden dimension buffer for ffn
    float *hb2;         // hidden dimension buffer for ffn
    float *q;           // query
    float *k;           // key
    float *v;           // value
    float *att;         // attention scores
    float *logits;      // output logits
    float* key_cache;   // KV cache
    float* value_cache; // KV cache
} RunState;

typedef struct {
    Config config;
    TransformerWeightsOffsets offsets;
    RunState state;
    FILE* file_handle;  // Keep file open for streaming direct from flash
    size_t file_size;
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    if (p->n_heads == 0) {
        fprintf(stderr, "Error: n_heads is zero!\n");
        return;
    }
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    
    // Allocate runtime activation state in internal SRAM
    s->x = (float*)calloc(p->dim, sizeof(float));
    s->xb = (float*)calloc(p->dim, sizeof(float));
    s->xb2 = (float*)calloc(p->dim, sizeof(float));
    s->hb = (float*)calloc(p->hidden_dim, sizeof(float));
    s->hb2 = (float*)calloc(p->hidden_dim, sizeof(float));
    s->q = (float*)calloc(p->dim, sizeof(float));
    s->key_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->value_cache = (float*)calloc(p->n_layers * p->seq_len * kv_dim, sizeof(float));
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    
    // Safety Guard: Stop execution immediately if heap space is exhausted
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "\n❌ FATAL: Out of SRAM Heap memory! Try lowering max seq_len.\n");
        return;
    }
}

void free_run_state(RunState* s) {
    if (s->x) free(s->x);
    if (s->xb) free(s->xb);
    if (s->xb2) free(s->xb2);
    if (s->hb) free(s->hb);
    if (s->hb2) free(s->hb2);
    if (s->q) free(s->q);
    if (s->att) free(s->att);
    if (s->logits) free(s->logits);
    if (s->key_cache) free(s->key_cache);
    if (s->value_cache) free(s->value_cache);
}

void calculate_weight_offsets(TransformerWeightsOffsets *w, Config* p, int shared_weights) {
    int head_size = p->dim / p->n_heads;
    unsigned long long n_layers = p->n_layers;
    size_t ptr = sizeof(Config); 

    w->token_embedding_table = ptr;
    ptr += (size_t)p->vocab_size * p->dim * sizeof(float);
    w->rms_att_weight = ptr;
    ptr += n_layers * p->dim * sizeof(float);
    w->wq = ptr;
    ptr += n_layers * p->dim * (p->n_heads * head_size) * sizeof(float);
    w->wk = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(float);
    w->wv = ptr;
    ptr += n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(float);
    w->wo = ptr;
    ptr += n_layers * (p->n_heads * head_size) * p->dim * sizeof(float);
    w->rms_ffn_weight = ptr;
    ptr += n_layers * p->dim * sizeof(float);
    w->w1 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim * sizeof(float);
    w->w2 = ptr;
    ptr += n_layers * p->hidden_dim * p->dim * sizeof(float);
    w->w3 = ptr;
    ptr += n_layers * p->dim * p->hidden_dim * sizeof(float);
    w->rms_final_weight = ptr;
    ptr += p->dim * sizeof(float);
    ptr += (p->seq_len * head_size / 2) * sizeof(float);
    ptr += (p->seq_len * head_size / 2) * sizeof(float);
    w->wcls = shared_weights ? w->token_embedding_table : ptr;
}

FILE* open_file_with_fallback(const char* path, const char* mode) {
    FILE* file = fopen(path, mode);
    if (!file && path[0] == '/') {
        char alt_path[128];
        snprintf(alt_path, sizeof(alt_path), "/littlefs%s", path);
        file = fopen(alt_path, mode);
    }
    return file;
}

void build_transformer(Transformer *t, const char* checkpoint_path) {
    t->file_handle = open_file_with_fallback(checkpoint_path, "rb");
    if (!t->file_handle) { 
        fprintf(stderr, "Couldn't open model file: %s\n", checkpoint_path); 
        memset(&t->config, 0, sizeof(Config));
        return; 
    }
    
    if (fread(&t->config, sizeof(Config), 1, t->file_handle) != 1) { 
        fprintf(stderr, "Failed to read header config\n"); 
        fclose(t->file_handle); 
        memset(&t->config, 0, sizeof(Config));
        return; 
    }

    if (t->config.n_heads == 0 || t->config.dim == 0) {
        fprintf(stderr, "Invalid model header configuration!\n");
        fclose(t->file_handle);
        memset(&t->config, 0, sizeof(Config));
        return;
    }

    int shared_weights = t->config.vocab_size > 0 ? 1 : 0;
    t->config.vocab_size = abs(t->config.vocab_size);

    // --- CAP SEQUENCE LENGTH FOR NON-PSRAM ESP32 SRAM LIMITS ---
    if (t->config.seq_len > 64) {
        t->config.seq_len = 64; 
    }

    calculate_weight_offsets(&t->offsets, &t->config, shared_weights);
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    if (t->file_handle) fclose(t->file_handle);
    free_run_state(&t->state);
}

void stream_read(FILE* file, size_t offset, void* dest, size_t bytes) {
    fseek(file, offset, SEEK_SET);
    fread(dest, 1, bytes, file);
}

void matmul_stream(FILE* file, size_t weight_offset, float* xout, float* x, int n, int d) {
    float row_buffer[256]; 
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        size_t row_offset = weight_offset + (size_t)i * n * sizeof(float);
        
        int processed = 0;
        while (processed < n) {
            int chunk = (n - processed > 256) ? 256 : (n - processed);
            stream_read(file, row_offset + processed * sizeof(float), row_buffer, chunk * sizeof(float));
            for (int j = 0; j < chunk; j++) {
                val += row_buffer[j] * x[processed + j];
            }
            processed += chunk;
        }
        xout[i] = val;
    }
}

// ----------------------------------------------------------------------------
// Neural net blocks

void rmsnorm_stream(FILE* file, size_t weight_offset, float* o, float* x, int size) {
    float weight_buffer[256];
    float ss = 0.0f;
    for (int j = 0; j < size; j++) {
        ss += x[j] * x[j];
    }
    ss /= size;
    ss += 1e-5f;
    ss = 1.0f / sqrtf(ss);

    int processed = 0;
    while (processed < size) {
        int chunk = (size - processed > 256) ? 256 : (size - processed);
        stream_read(file, weight_offset + processed * sizeof(float), weight_buffer, chunk * sizeof(float));
        for (int j = 0; j < chunk; j++) {
            o[processed + j] = weight_buffer[j] * (ss * x[processed + j]);
        }
        processed += chunk;
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

float* forward(Transformer* transformer, int token, int pos) {
    Config* p = &transformer->config;
    TransformerWeightsOffsets* w = &transformer->offsets;
    RunState* s = &transformer->state;
    FILE* file = transformer->file_handle;
    
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; 
    int hidden_dim = p->hidden_dim;
    int head_size = dim / p->n_heads;

    size_t embed_offset = w->token_embedding_table + (size_t)token * dim * sizeof(float);
    stream_read(file, embed_offset, x, dim * sizeof(float));

    for (unsigned long long l = 0; l < p->n_layers; l++) {
        rmsnorm_stream(file, w->rms_att_weight + l * dim * sizeof(float), s->xb, x, dim);

        int loff = l * p->seq_len * kv_dim;
        s->k = s->key_cache + loff + pos * kv_dim;
        s->v = s->value_cache + loff + pos * kv_dim;

        size_t q_offset = w->wq + l * dim * dim * sizeof(float);
        size_t k_offset = w->wk + l * dim * kv_dim * sizeof(float);
        size_t v_offset = w->wv + l * dim * kv_dim * sizeof(float);

        matmul_stream(file, q_offset, s->q, s->xb, dim, dim);
        matmul_stream(file, k_offset, s->k, s->xb, dim, kv_dim);
        matmul_stream(file, v_offset, s->v, s->xb, dim, kv_dim);

        for (int i = 0; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            int rotn = i < kv_dim ? 2 : 1;
            for (int v = 0; v < rotn; v++) {
                float* vec = v == 0 ? s->q : s->k;
                float v0 = vec[i];
                float v1 = vec[i+1];
                vec[i]   = v0 * fcr - v1 * fci;
                vec[i+1] = v0 * fci + v1 * fcr;
            }
        }

        for (int h = 0; h < p->n_heads; h++) {
            float* q = s->q + h * head_size;
            float* att = s->att + h * p->seq_len;
            for (int t = 0; t <= pos; t++) {
                float* k = s->key_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i] * k[i];
                }
                score /= sqrtf(head_size);
                att[t] = score;
            }

            softmax(att, pos + 1);

            float* xb = s->xb + h * head_size;
            memset(xb, 0, head_size * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float* v = s->value_cache + loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t];
                for (int i = 0; i < head_size; i++) {
                    xb[i] += a * v[i];
                }
            }
        }

        size_t o_offset = w->wo + l * dim * dim * sizeof(float);
        matmul_stream(file, o_offset, s->xb2, s->xb, dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb2[i];
        }

        rmsnorm_stream(file, w->rms_ffn_weight + l * dim * sizeof(float), s->xb, x, dim);

        size_t w1_offset = w->w1 + l * dim * hidden_dim * sizeof(float);
        size_t w2_offset = w->w2 + l * hidden_dim * dim * sizeof(float);
        size_t w3_offset = w->w3 + l * dim * hidden_dim * sizeof(float);

        matmul_stream(file, w1_offset, s->hb, s->xb, dim, hidden_dim);
        matmul_stream(file, w3_offset, s->hb2, s->xb, dim, hidden_dim);

        for (int i = 0; i < hidden_dim; i++) {
            float val = s->hb[i];
            val *= (1.0f / (1.0f + expf(-val)));
            val *= s->hb2[i];
            s->hb[i] = val;
        }

        matmul_stream(file, w2_offset, s->xb, s->hb, hidden_dim, dim);

        for (int i = 0; i < dim; i++) {
            x[i] += s->xb[i];
        }
    }

    rmsnorm_stream(file, w->rms_final_weight, x, x, dim);
    matmul_stream(file, w->wcls, s->logits, x, p->dim, p->vocab_size);
    return s->logits;
}

// ----------------------------------------------------------------------------
// Tokenizer implementation

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, const char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char**)heap_alloc(vocab_size * sizeof(char*));
    t->vocab_scores = (float*)heap_alloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;

    if (!t->vocab || !t->vocab_scores) {
        fprintf(stderr, "Failed to allocate memory for tokenizer structures.\n");
        return;
    }

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    FILE *file = open_file_with_fallback(tokenizer_path, "rb");
    if (!file) { 
        fprintf(stderr, "couldn't load tokenizer from %s\n", tokenizer_path); 
        return; 
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { 
        fprintf(stderr, "failed tokenizer read\n"); 
        fclose(file); 
        return; 
    }

    int len;
    for (int i = 0; i < vocab_size; i++) {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1) break;
        if (fread(&len, sizeof(int), 1, file) != 1) break;
        t->vocab[i] = (char *)heap_alloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) break;
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

void free_tokenizer(Tokenizer* t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) { 
            if (t->vocab[i]) free(t->vocab[i]); 
        }
        free(t->vocab);
    }
    if (t->vocab_scores) free(t->vocab_scores);
    if (t->sorted_vocab) free(t->sorted_vocab);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    if (piece == NULL || piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = { .str = str };
    TokenIndex *res = (TokenIndex*)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer* t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
    if (text == NULL) { fprintf(stderr, "cannot encode NULL text\n"); return; }

    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex*)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }

    char* str_buffer = (char*)malloc((t->max_token_length * 2 + 1 + 2) * sizeof(char));
    size_t str_len = 0;
    *n_tokens = 0;

    if (bos) tokens[(*n_tokens)++] = 1;

    if (text[0] != '\0') {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            str_len = 0;
        }

        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';

        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            tokens[(*n_tokens)++] = id;
        } else {
            for (int i = 0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0;
    }

    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--;
    }

    if (eos) tokens[(*n_tokens)++] = 2;

    free(str_buffer);
}

// ----------------------------------------------------------------------------
// Sampler implementation

typedef struct {
    float prob;
    int index;
} ProbIndex;

typedef struct {
    int vocab_size;
    ProbIndex* probindex;
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1;
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    int n0 = 0;
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1;
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break;
        }
    }

    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index;
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    sampler->probindex = (ProbIndex*)malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    if (sampler->probindex) free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}

float random_f32(unsigned long long *state) {
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    int next;
    if (sampler->temperature == 0.0f) {
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        for (int q = 0; q < sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        softmax(logits, sampler->vocab_size);
        float coin = random_f32(&sampler->rng_state);
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// Generation utility functions

long time_in_ms() {
#ifdef ESP32
    return (long)(xTaskGetTickCount() * portTICK_PERIOD_MS);
#else
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
#endif
}

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    if (transformer->config.dim == 0 || transformer->config.n_heads == 0 || !transformer->state.x) {
        fprintf(stderr, "Cannot generate, invalid Transformer allocation.\n");
        return;
    }

    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt) + 3) * sizeof(int));
    encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "Expected at least 1 prompt token\n");
        free(prompt_tokens);
        return;
    }

    long start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    
    // Do not exceed configured context sequence window
    if (steps > transformer->config.seq_len) {
        steps = transformer->config.seq_len;
    }

    while (pos < steps) {
        float* logits = forward(transformer, token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sample(sampler, logits);
        }
        pos++;

        if (next == 1) { break; }

        char* piece = decode(tokenizer, token, next);
        safe_printf(piece);
        fflush(stdout);
        token = next;

        if (start == 0) { start = time_in_ms(); }

#ifdef ESP32
        vTaskDelay(1);
#endif
    }
    printf("\n");

    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos - 1) / (double)(end - start) * 1000);
    }

    free(prompt_tokens);
}

void run_llama(const char* checkpoint_path, const char* tokenizer_path, float temperature, float topp, int steps, const char* prompt) {
    Transformer transformer;
    build_transformer(&transformer, checkpoint_path);
    if (transformer.config.dim == 0 || transformer.config.n_heads == 0 || !transformer.state.x) {
        printf("Failed to load model architecture or allocate state memory.\n");
        if (transformer.file_handle) fclose(transformer.file_handle);
        return;
    }

    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, transformer.config.vocab_size);

    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, (unsigned long long)time(NULL));

    generate(&transformer, &tokenizer, &sampler, (char*)prompt, steps);

    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
}
