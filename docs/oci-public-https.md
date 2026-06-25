# OCI Public HTTPS

Run this once on the Oracle Linux VM after SSH login:

```bash
bash scripts/open-caddy-ports-firewalld.sh
```

If you do not copy the repository to the VM, run the equivalent commands directly:

```bash
sudo systemctl enable --now firewalld
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --permanent --add-service=https
sudo firewall-cmd --permanent --remove-port=8000/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-services
```

Add the OCI Network Security Group ingress rule for `ig-quick-action-NSG` by
passing its OCID. Check existing rules first to avoid duplicates:

```bash
oci network nsg rules list \
  --network-security-group-id "<NSG_OCID_FOR_IG_QUICK_ACTION_NSG>"

oci network nsg rules add \
  --network-security-group-id "<NSG_OCID_FOR_IG_QUICK_ACTION_NSG>" \
  --security-rules '[
    {
      "direction": "INGRESS",
      "protocol": "6",
      "source": "0.0.0.0/0",
      "sourceType": "CIDR_BLOCK",
      "tcpOptions": {"destinationPortRange": {"min": 80, "max": 80}},
      "description": "Allow public HTTP for Caddy redirects and ACME"
    },
    {
      "direction": "INGRESS",
      "protocol": "6",
      "source": "0.0.0.0/0",
      "sourceType": "CIDR_BLOCK",
      "tcpOptions": {"destinationPortRange": {"min": 443, "max": 443}},
      "description": "Allow public HTTPS for Medic"
    }
  ]'
```

After `https://<MEDIC_DOMAIN>/healthz` returns HTTP `204`, remove the old TCP
`8000` ingress rule from the NSG.
