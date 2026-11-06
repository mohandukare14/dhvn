#ifndef DHVN_TRAIN_H
#define DHVN_TRAIN_H

#include "dhvn_core.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    int dim_c;
    int num_inputs; // 2 * DHVN_MAX_HARMONICS
    
    // Trainable parameters
    float* W_readout;  // vocab_size x dim_c
    float* W_charge;   // dim_c x num_inputs (points to sys->W_charge)
    float* W_gate;     // 1 x num_inputs (points to sys->W_gate)
    
    // Gradients
    float* grad_W_readout;
    float* grad_W_charge;
    float* grad_W_gate;
    
    // Adam optimizer states (first and second moments)
    float* m_W_readout;
    float* v_W_readout;
    float* m_W_charge;
    float* v_W_charge;
    float* m_W_gate;
    float* v_W_gate;
    
    int t_step; // Adam step counter
} DHVNTrainer;

DHVNTrainer* dhvn_trainer_create(DHVNSystem* sys, int vocab_size);
void dhvn_trainer_free(DHVNTrainer* trainer);

// Single training step over a sequence of length T
// Computes cross-entropy loss, adjoint backward pass, and applies Adam optimizer
float dhvn_train_sequence(
    DHVNSystem* sys,
    DHVNTrainer* trainer,
    const unsigned char* in_bytes,
    const unsigned char* target_bytes,
    int length,
    float lr,
    float dt
);

// Forward evaluation / inference: returns top predicted byte and cross-entropy loss
int dhvn_predict_next(
    DHVNSystem* sys,
    DHVNTrainer* trainer,
    unsigned char in_byte,
    float dt,
    float* out_prob
);

// Full distribution prediction for Temperature and Top-p (Nucleus) sampling
void dhvn_predict_distribution(
    DHVNSystem* sys,
    DHVNTrainer* trainer,
    unsigned char in_byte,
    float dt,
    float* out_probs
);

#ifdef __cplusplus
}
#endif

#endif // DHVN_TRAIN_H

