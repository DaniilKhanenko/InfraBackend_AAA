import asyncio
import aiohttp
import time
import numpy as np

URL_EMBED = "http://localhost:8000/embed"
URL_METRICS = "http://localhost:8000/sys_metrics"
TEXT = "Это тестовое сообщение для нашей маленькой модельки, чтобы я мог замерить метрики сервера"

N_REQUESTS = 200
USERS = 10

async def fetch_embed(session, sem):
    async with sem:
        start_time = time.time()
        async with session.post(URL_EMBED, json={"text": TEXT}) as response:
            res = await response.json()
            latency = (time.time() - start_time) * 1000
            return latency, res["inference_time_ms"]

async def run_benchmark():    
    async with aiohttp.ClientSession() as session:
        async with session.get(URL_METRICS) as resp:
            mem_before = (await resp.json())["ram_mb"]
        sem = asyncio.Semaphore(USERS)
        start_time = time.time()
        tasks =[fetch_embed(session, sem) for _ in range(N_REQUESTS)]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        async with session.get(URL_METRICS) as resp:
            mem_after = (await resp.json())["ram_mb"]

    latencies = [r[0] for r in results]
    inference_times = [r[1] for r in results]

    rps = N_REQUESTS / total_time
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_inf = np.mean(inference_times)
    
    print(f"Throughput (RPS):       {rps:.2f} запросов/сек")
    print(f"Latency P50 (Медиана):  {p50:.2f} мс")
    print(f"Latency P95:            {p95:.2f} мс")
    print(f"Latency P99:            {p99:.2f} мс")
    print(f"Mean Inference Time:    {mean_inf:.2f} мс")
    print(f"RAM до нагрузки:        {mem_before:.2f} MB")
    print(f"RAM после нагрузки:     {mem_after:.2f} MB")

if __name__ == "__main__":
    asyncio.run(run_benchmark())