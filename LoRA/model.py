import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import (LoraConfig, TaskType, get_peft_config,  # <-- Add this
                  get_peft_model)
from transformers import AutoConfig, AutoModel


def kl_div(p, q):
    return (p * ((p + 1e-5).log() - (q + 1e-5).log())).sum(-1)

class NERModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=args.num_class)
        self.base_model = AutoModel.from_pretrained(args.model_name_or_path)
        
        # Correct LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,  # Use FEATURE_EXTRACTION
            inference_mode=False,
            r=16,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["query", "value"],  # Target attention layers
            modules_to_save=["classifier"]      # Include classifier if needed
        )
        
        self.model = get_peft_model(self.base_model, peft_config)
        self.model.print_trainable_parameters()
        
        self.dropout = nn.Dropout(args.dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, args.num_class)
        self.loss_fnt = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        # Explicitly pass only necessary arguments to avoid kwargs leakage
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )
        h = outputs[0]
        h = self.dropout(h)
        logits = self.classifier(h)
        logits = logits.view(-1, self.args.num_class)
        outputs = (logits,)
        if labels is not None:
            labels = labels.view(-1)
            loss = self.loss_fnt(logits, labels)
            outputs = (loss,) + outputs
        return outputs


class NLLModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.models = nn.ModuleList()
        self.device = [i % args.n_gpu for i in range(args.n_model)]
        self.loss_fnt = nn.CrossEntropyLoss()
        for i in range(args.n_model):
            model = NERModel(args)
            model.to(self.device[i])
            self.models.append(model)

    def forward(self, input_ids, attention_mask, labels=None):
        if labels is None:
            return self.models[0](input_ids=input_ids,
                                  attention_mask=attention_mask,
                                  )
        else:
            num_models = len(self.models)
            outputs = []
            for i in range(num_models):
                output = self.models[i](
                    input_ids=input_ids.to(self.device[i]),
                    attention_mask=attention_mask.to(self.device[i]),
                    labels=labels.to(self.device[i]) if labels is not None else None,
                )
                output = tuple([o.to(0) for o in output])
                outputs.append(output)
            model_output = outputs[0]
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [output[1] for output in outputs]
            probs = [F.softmax(logit, dim=-1) for logit in logits]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            mask = (labels.view(-1) != -1).to(logits[0])
            reg_loss = sum([kl_div(avg_prob, prob) * mask for prob in probs]) / num_models
            reg_loss = reg_loss.sum() / (mask.sum() + 1e-3)
            loss = loss + self.args.alpha_t * reg_loss
            model_output = (loss,) + model_output[1:]
        return model_output
