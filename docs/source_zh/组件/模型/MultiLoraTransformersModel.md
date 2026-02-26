# MultiLoraTransformersModel

这个模型继承了TransformersModel，除提供了相同功能外，还提供了分时运行多个lora的能力，主要用于多租户训练。

```python
class MultiLoraTransformersModel:

    def __init__(self,  # noqa
                 model_cls = AutoModelForCausalLM,
                 model_id: Optional[str] = None,
                 config: Optional[PretrainedConfig] = None,
                 device_mesh: Optional[DeviceMesh] = None,
                 mixed_precision: Literal['no', 'fp8', 'fp16', 'bf16'] = 'bf16',
                 strategy: Literal['accelerate', 'native_fsdp'] = 'accelerate',
                 fsdp_config: Dict[str, Any] = None,
                 grad_scaler_config: Dict[str, Any] = None,
                 max_loras: int = 5,
                 max_r: int = 32,
                 max_length: int = 8192,
                 **kwargs):
        ...

    ...
```

除了和基类相同的参数外，本类提供了几个额外参数用于多lora配置：
- max_loras: 最大lora的数量
- max_r: 最大的lora rank
- max_length: 最大的支持训练长度
- strategy: 多租户 LoRA 在 FSDP 场景下仅支持 `native_fsdp`
- fsdp_config: `native_fsdp` 使用的配置

之所以存在max_loras和max_r参数，是因为twinkle的多lora技术方案是在DDP wrap之前增加lora到`max_loras`个，防止后添加的lora无法接受DDP的管理。
正因如此，用户的r必须要小于等于max_r的配置，在实际训练时仅会使用lora的部分rank参与计算。

当 `device_mesh.fsdp_world_size > 1` 时，MultiLoraTransformersModel 会强制使用 `native_fsdp`，
并保持“预分配槽位 + 运行时租户复用”的多租户 LoRA 机制。

MultiLoraTransformersModel支持`@remote_class`注解，并且支持device_mesh，这意味着它可以运行在ray的worker中。
