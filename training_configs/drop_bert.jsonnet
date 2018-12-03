{
    "dataset_reader": {
        "type": "drop",
        "token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            },
            "bert": {
                "type": "bert-pretrained",
                "pretrained_model": "bert-base-uncased",
                "use_starting_offsets": true
            },
            "token_characters": {
                "type": "characters",
                "min_padding_length": 5
            }
        },
        "passage_length_limit": 350,
        "question_length_limit": 50,
        "passage_length_limit_for_evaluation": 350,
        "question_length_limit_for_evaluation": 50
    },
    "vocabulary": {
        "min_count": {
            "token_characters": 200
        },
        "pretrained_files": {
            "tokens": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz"
        },
        "only_include_pretrained_words": true
    },
    "train_data_path": "https://s3-us-west-2.amazonaws.com/yizhongw-dev/sparc/drop_dataset_train.json",
    "validation_data_path": "https://s3-us-west-2.amazonaws.com/yizhongw-dev/sparc/drop_dataset_dev.json",
    "model": {
        "type": "qanet_for_drop",
        "text_field_embedder": {
            "allow_unmatched_keys": true,
            "embedder_to_indexer_map": {
                "bert": ["bert", "bert-offsets"],
                "tokens": ["tokens"],
                "token_characters": ["token_characters"],
            },
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "pretrained_file": "https://s3-us-west-2.amazonaws.com/allennlp/datasets/glove/glove.6B.100d.txt.gz",
                    "embedding_dim": 100,
                    "trainable": false
                },
                "bert": {
                    "type": "bert-pretrained",
                    "pretrained_model": "bert-base-uncased"
                },
                "token_characters": {
                    "type": "character_encoding",
                    "embedding": {
                        "embedding_dim": 64
                    },
                    "encoder": {
                        "type": "cnn",
                        "embedding_dim": 64,
                        "num_filters": 200,
                        "ngram_filter_sizes": [
                            5
                        ]
                    }
                }
            }
        },
        "num_highway_layers": 2,
        "phrase_layer": {
            "type": "qanet_encoder",
            "input_dim": 128,
            "hidden_dim": 128,
            "attention_projection_dim": 128,
            "feedforward_hidden_dim": 128,
            "num_blocks": 1,
            "num_convs_per_block": 4,
            "conv_kernel_size": 7,
            "num_attention_heads": 8,
            "dropout_prob": 0.1,
            "layer_dropout_undecayed_prob": 0.1,
            "attention_dropout_prob": 0
        },
        "matrix_attention_layer": {
            "type": "linear",
            "tensor_1_dim": 128,
            "tensor_2_dim": 128,
            "combination": "x,y,x*y"
        },
        "modeling_layer": {
            "type": "qanet_encoder",
            "input_dim": 128,
            "hidden_dim": 128,
            "attention_projection_dim": 128,
            "feedforward_hidden_dim": 128,
            "num_blocks": 6,
            "num_convs_per_block": 2,
            "conv_kernel_size": 5,
            "num_attention_heads": 8,
            "dropout_prob": 0.1,
            "layer_dropout_undecayed_prob": 0.1,
            "attention_dropout_prob": 0
        },
        "dropout_prob": 0.1,
        "regularizer": [
            [
                ".*",
                {
                    "type": "l2",
                    "alpha": 1e-07
                }
            ]
        ]
    },
    "iterator": {
        "type": "bucket",
        "sorting_keys": [
            [
                "passage",
                "num_tokens"
            ],
            [
                "question",
                "num_tokens"
            ]
        ],
        "batch_size": 16,
        "max_instances_in_memory": 600
    },
    "trainer": {
        "type": "ema_trainer",
        "num_epochs": 50,
        "grad_norm": 5,
        "patience": 10,
        "validation_metric": "+f1",
        "cuda_device": 0,
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "betas": [
                0.8,
                0.999
            ],
            "eps": 1e-07
        },
        "exponential_moving_average_decay": 0.9999
    }
}
