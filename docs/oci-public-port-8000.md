# OCI Public Port 8000

Run this once on the Oracle Linux VM after SSH login:

```bash
bash scripts/open-app-port-8000-firewalld.sh
```

If you do not copy the repository to the VM, run the equivalent commands directly:

```bash
sudo systemctl enable --now firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --reload
sudo firewall-cmd --list-ports
```

Add the OCI Network Security Group ingress rule for `ig-quick-action-NSG` by
passing its OCID:

```bash
oci network nsg rules add \
  --network-security-group-id "<NSG_OCID_FOR_IG_QUICK_ACTION_NSG>" \
  --security-rules '[{"direction":"INGRESS","protocol":"6","source":"0.0.0.0/0","sourceType":"CIDR_BLOCK","tcpOptions":{"destinationPortRange":{"min":8000,"max":8000}},"description":"Allow public HTTP access to Medic dashboard on TCP 8000"}]'
```
