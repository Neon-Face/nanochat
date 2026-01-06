"""
IPv6 Verifier (Scanner) for 6GPT.
"""
import subprocess
import concurrent.futures
import random

class IPv6Scanner:
    def __init__(self, use_mock=False):
        self.use_mock = use_mock
        # 模拟模式：用于调试代码逻辑，不真的发包
        if use_mock:
            print("WARNING: Scanner is in MOCK mode.")

    def verify_batch(self, ip_list):
        """
        输入: List[str] (IP地址列表)
        输出: List[float] (1.0 = Active, 0.0 = Dead)
        """
        if self.use_mock:
            # 假装 10% 的地址是活的
            return [1.0 if random.random() < 0.1 else 0.0 for _ in ip_list]

        # 并行 Ping (Mac M3 性能很强，可以开多线程)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            future_to_ip = {executor.submit(self._ping_single, ip): ip for ip in ip_list}
            
            # 保持顺序很重要！需要按输入列表的顺序返回结果
            # 所以我们创建一个 map
            ip_status = {}
            for future in concurrent.futures.as_completed(future_to_ip):
                ip = future_to_ip[future]
                is_active = future.result()
                ip_status[ip] = 1.0 if is_active else 0.0
        
        # 按原始顺序返回
        return [ip_status.get(ip, 0.0) for ip in ip_list]

    def _ping_single(self, ip):
        # Mac/Linux 通用的 Ping6 命令
        # -c 1: 发送1个包
        # -W 500: 等待 500ms (Mac ping6 的参数可能是 -i 或 -W，视版本而定)
        # Linux通常是 -W 1 (秒)。Mac 可能是 -t 1。
        # 为了兼容，我们用 subprocess 的 timeout 参数控制
        try:
            # 简单的 Ping 测试
            # 警告：很多 ISP 会屏蔽 Ping，但在原型阶段这是最好的验证方式
            subprocess.check_call(
                ["ping6", "-c", "1", ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=0.5 # 0.5秒超时
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False

# 测试代码
if __name__ == "__main__":
    scanner = IPv6Scanner(use_mock=False)
    test_ips = ["2400:3200::1", "2001:4860:4860::8888", "2001:db8::1"] # Aliyun DNS, Google DNS, Fake
    print(f"Testing IPs: {test_ips}")
    results = scanner.verify_batch(test_ips)
    print(f"Results: {results}")