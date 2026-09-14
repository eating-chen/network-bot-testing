# `data_rca/taxonomy.py`

## 角色

用 deterministic keyword/category rules 將 source-native labels 映射成 `network_domain` 與 `protocol`。

## Functions

### `nika_domain(fault_type)`

- Input：NIKA fault type string，例如 `bgp_asn_misconfig`。
- Output：`tuple[str, str]`，例如 `("routing", "bgp")`。
- Mapping priority：BGP → OSPF → DNS → DHCP → ARP → ACL/attack → K8s → P4 → generic IP routing → interface → system。
- 注意：第一個 keyword match 立即 return。

### `anta_domain(categories)`

- Input：ANTA categories list，例如 `["bgp"]` 或 `["vxlan"]`。
- Output：`(network_domain, protocol)`。
- Routing protocols：BGP、OSPF、ISIS、BFD。
- Layer 2：MLAG、VXLAN、EVPN、STP、VLAN。
- 另映射 interfaces/connectivity、security/AAA、services/SNMP/logging、generic routing。
- 未知 category 原樣成為 domain；空 categories fallback `("system", "")`。

## Input / output examples

```python
nika_domain("bgp_asn_misconfig")
# -> ("routing", "bgp")

nika_domain("dns_service_down")
# -> ("services", "dns")

anta_domain(["bgp"])
# -> ("routing", "bgp")

anta_domain(["vxlan"])
# -> ("layer2", "vxlan")

anta_domain([])
# -> ("system", "")
```
