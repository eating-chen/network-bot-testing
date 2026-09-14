"""Small deterministic mappings used by coverage reports and converters."""


def nika_domain(fault_type: str) -> tuple[str, str]:
    name = fault_type.lower()
    if "bgp" in name:
        return "routing", "bgp"
    if "ospf" in name:
        return "routing", "ospf"
    if "dns" in name:
        return "services", "dns"
    if "dhcp" in name:
        return "services", "dhcp"
    if "arp" in name:
        return "layer2", "arp"
    if "acl" in name or "attack" in name:
        return "security", "acl" if "acl" in name else ""
    if "k8s" in name or "kubernetes" in name:
        return "orchestration", "kubernetes"
    if "p4" in name:
        return "data_plane", "p4"
    if any(word in name for word in ("route", "gateway", "subnet", "ip_")):
        return "routing", "ip"
    if any(word in name for word in ("link", "interface", "mtu")):
        return "interfaces", "ethernet"
    return "system", ""


def anta_domain(categories: list[str]) -> tuple[str, str]:
    normalized = [value.lower().replace(" ", "_") for value in categories]
    protocols = {
        "bgp": "bgp",
        "ospf": "ospf",
        "isis": "isis",
        "mlag": "mlag",
        "vxlan": "vxlan",
        "evpn": "evpn",
        "stp": "stp",
        "vlan": "vlan",
        "bfd": "bfd",
    }
    for category, protocol in protocols.items():
        if category in normalized:
            domain = "routing" if category in {"bgp", "ospf", "isis", "bfd"} else "layer2"
            return domain, protocol
    if any(value in normalized for value in ("interfaces", "connectivity")):
        return "interfaces", "ethernet"
    if any(value in normalized for value in ("security", "aaa")):
        return "security", ""
    if any(value in normalized for value in ("services", "snmp", "logging")):
        return "services", ""
    if any(value in normalized for value in ("routing", "path-selection", "path_selection")):
        return "routing", "ip"
    return (normalized[0] if normalized else "system"), ""
