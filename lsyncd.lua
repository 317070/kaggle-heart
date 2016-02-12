settings {
   nodaemon   = true
}
sync {
    default.rsyncssh,
    source  = ".",
    targetdir  = "/mnt/storage/users/lpigou/kaggle-heart",
    host="lpigou@paard",
    exclude={"paths.yaml", ".*", "lsyncd.lua", "lsyncd.log", "lsyncd.status"},
    delete=false,
    delay=0.5,
    rsync = {
        archive = true,
        verbose=true,
        perms=false,
        times=false,
        _extra={}
    },
}