"""Rebuild only workspace packages once per checker and Cargo target directory."""
from pathlib import Path
import subprocess,json,re,shutil,hashlib
_cleaned=set()
def ensure_project_build(env, workspace, out):
    target=str(Path(env['CARGO_TARGET_DIR']).resolve())
    key=(str(Path(workspace).resolve()),target)
    if key in _cleaned:return
    suffix='release' if Path(target).name=='check-release' else 'dev'
    command=['cargo','clean']
    for package in ['bitnet-core','bitnet-server','rbitnet-cli','rbitnet-proxy','rbitnet-runner']:
        command.extend(['-p',package])
    with (Path(out)/('project-artifact-clean-'+suffix+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run(command,cwd=workspace,env=env,stdout=log,stderr=subprocess.STDOUT)
    assert result.returncode==0,('workspace artifact clean failed',target,result.returncode)
    # Cargo's package-specific clean selects the current path package ID.
    # Older same-name path packages can leave a different hash in this shared
    # target. Remove every project hash, preserving third-party dependencies.
    base=Path(__file__).resolve().parent
    selected=Path(target)
    assert selected in [(base/'check-build').resolve(),(base/'check-release').resolve()]
    package=re.compile(r'^(?:bitnet-core|bitnet-server|rbitnet-cli|rbitnet-proxy|rbitnet-runner)-[0-9a-z]+$')
    artifact=re.compile(r'^(?:lib)?(?:bitnet_core|bitnet_server|rbitnet|rbitnet_cli|rbitnet_proxy|rbitnet_runner)(?:-[0-9a-z]+)?\.(?:rmeta|rlib|d|exe|pdb|dll|lib|exp|o|obj)$')
    paths=[]
    for profile in ['debug','release']:
        folder=selected/profile
        for location in [folder,folder/'deps']:
            if location.exists():paths.extend(p for p in location.iterdir() if p.is_file() and artifact.fullmatch(p.name))
        for location in [folder/'.fingerprint',folder/'build']:
            if location.exists():paths.extend(p for p in location.iterdir() if p.is_dir() and package.fullmatch(p.name))
    # Validate every resolved target before any removal; reject filesystem links.
    for path in paths:
        assert not path.is_symlink() and path.resolve().is_relative_to(selected),path
    receipt=[]
    for path in paths:
        item={'path':path.relative_to(selected).as_posix(),'directory':path.is_dir()}
        if path.is_file():
            item['bytes']=path.stat().st_size
            if path.suffix=='.rmeta':item['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            path.unlink()
        else:shutil.rmtree(path)
        receipt.append(item)
    (Path(out)/('all-project-artifact-purge-'+suffix+'.json')).write_text(json.dumps({'workspace':str(workspace),'target':target,'removed':receipt,'external_dependencies_preserved':True},indent=2),encoding='utf-8')
    _cleaned.add(key)
