import asyncio

from twinkle.sampler.vllm_sampler.vllm_engine import VLLMEngine


def test_concurrent_lora_requests_share_one_load_task():
    async def run():
        engine = VLLMEngine.__new__(VLLMEngine)
        engine._lora_request_cache = {}
        engine._lora_load_tasks = {}
        request = object()
        load_count = 0

        async def load_lora(_path):
            nonlocal load_count
            load_count += 1
            await asyncio.sleep(.01)
            return request

        engine._load_lora = load_lora
        results = await asyncio.gather(*(engine._get_or_load_lora('/adapter') for _ in range(8)))

        assert load_count == 1
        assert results == [request] * 8
        assert engine._lora_request_cache == {'/adapter': request}
        assert engine._lora_load_tasks == {}

    asyncio.run(run())
