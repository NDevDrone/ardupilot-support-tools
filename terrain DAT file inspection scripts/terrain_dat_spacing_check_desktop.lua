#!/usr/bin/env lua
-- Check ArduPilot terrain .DAT tile spacing and block integrity.
--
-- Usage:
--   lua terrain_dat_spacing_check_desktop.lua /path/to/NxxExxx.DAT [expected_spacing_m]

local TERRAIN_GRID_FORMAT_VERSION = 1
local IO_BLOCK_SIZE = 2048
local IO_BLOCK_DATA_SIZE = 1821

local function usage()
    io.stderr:write("Usage: lua terrain_dat_spacing_check_desktop.lua <tile.DAT> [expected_spacing_m]\n")
    os.exit(2)
end

local function read_file(path)
    local fh, err = io.open(path, "rb")
    if not fh then
        io.stderr:write(string.format("Failed to open '%s': %s\n", path, tostring(err)))
        os.exit(2)
    end
    local data = fh:read("*a")
    fh:close()
    return data
end

local function u16_le(buf, p)
    local b1, b2 = buf:byte(p, p + 1)
    if not b2 then
        return nil
    end
    return b1 | (b2 << 8)
end

local function u32_le(buf, p)
    local b1, b2, b3, b4 = buf:byte(p, p + 3)
    if not b4 then
        return nil
    end
    return b1 | (b2 << 8) | (b3 << 16) | (b4 << 24)
end

local function i32_le(buf, p)
    local v = u32_le(buf, p)
    if not v then
        return nil
    end
    if v >= 0x80000000 then
        return v - 0x100000000
    end
    return v
end

local function u64_all_zero(buf, p)
    local b1, b2, b3, b4, b5, b6, b7, b8 = buf:byte(p, p + 7)
    if not b8 then
        return true
    end
    return (b1 | b2 | b3 | b4 | b5 | b6 | b7 | b8) == 0
end

local function crc16_xmodem(data)
    local crc = 0
    for i = 1, #data do
        crc = crc ~ (data:byte(i) << 8)
        for _ = 1, 8 do
            if (crc & 0x8000) ~= 0 then
                crc = ((crc << 1) ~ 0x1021) & 0xFFFF
            else
                crc = (crc << 1) & 0xFFFF
            end
        end
    end
    return crc
end

if not arg or not arg[1] then
    usage()
end

local dat_path = arg[1]
if dat_path:sub(-3) == ".gz" then
    io.stderr:write("Input looks gzip-compressed (.gz). Please decompress to .DAT first.\n")
    os.exit(2)
end

local expected_spacing = nil
if arg[2] ~= nil then
    expected_spacing = tonumber(arg[2])
    if not expected_spacing then
        io.stderr:write(string.format("Invalid expected spacing '%s'\n", tostring(arg[2])))
        os.exit(2)
    end
end

local file_data = read_file(dat_path)
local file_size = #file_data
if file_size == 0 then
    io.stderr:write("File is empty.\n")
    os.exit(2)
end

local block_size = nil
if (file_size % IO_BLOCK_SIZE) == 0 then
    block_size = IO_BLOCK_SIZE
elseif (file_size % IO_BLOCK_DATA_SIZE) == 0 then
    block_size = IO_BLOCK_DATA_SIZE
else
    io.stderr:write(
        string.format(
            "Unsupported file length (%d). Expected multiple of %d (padded) or %d (unpadded).\n",
            file_size,
            IO_BLOCK_SIZE,
            IO_BLOCK_DATA_SIZE
        )
    )
    os.exit(2)
end

local total_blocks = 0
local valid_blocks = 0
local empty_blocks = 0
local header_errors = 0
local crc_errors = 0
local spacing_mismatch_blocks = 0
local spacing_counts = {}

for pos = 1, file_size, block_size do
    local block = file_data:sub(pos, pos + block_size - 1)
    if #block < IO_BLOCK_DATA_SIZE then
        header_errors = header_errors + 1
        break
    end

    total_blocks = total_blocks + 1

    local bitmap_is_zero = u64_all_zero(block, 1)
    local lat = i32_le(block, 9)
    local lon = i32_le(block, 13)
    local crc = u16_le(block, 17)
    local version = u16_le(block, 19)
    local spacing = u16_le(block, 21)

    if not lat or not lon or not crc or not version or not spacing then
        header_errors = header_errors + 1
        goto continue
    end

    if bitmap_is_zero and lat == 0 and lon == 0 and crc == 0 and version == 0 and spacing == 0 then
        empty_blocks = empty_blocks + 1
        goto continue
    end

    if version ~= TERRAIN_GRID_FORMAT_VERSION then
        header_errors = header_errors + 1
        goto continue
    end

    local crc_data = block:sub(1, IO_BLOCK_DATA_SIZE)
    crc_data = crc_data:sub(1, 16) .. "\0\0" .. crc_data:sub(19, IO_BLOCK_DATA_SIZE)
    local calc_crc = crc16_xmodem(crc_data)
    if calc_crc ~= crc then
        crc_errors = crc_errors + 1
        goto continue
    end

    valid_blocks = valid_blocks + 1
    spacing_counts[spacing] = (spacing_counts[spacing] or 0) + 1

    if expected_spacing and spacing ~= expected_spacing then
        spacing_mismatch_blocks = spacing_mismatch_blocks + 1
    end

    ::continue::
end

local unique_spacings = {}
for spacing, _ in pairs(spacing_counts) do
    table.insert(unique_spacings, spacing)
end
table.sort(unique_spacings)

print(string.format("File: %s", dat_path))
print(string.format("Size: %d bytes", file_size))
print(string.format("Block layout: %d-byte blocks", block_size))
print(string.format("Total blocks: %d", total_blocks))
print(string.format("Valid blocks: %d", valid_blocks))
print(string.format("Empty blocks: %d", empty_blocks))
print(string.format("Header errors: %d", header_errors))
print(string.format("CRC errors: %d", crc_errors))

if #unique_spacings == 0 then
    print("Spacing values: none (no valid terrain blocks)")
else
    print("Spacing values:")
    for _, spacing in ipairs(unique_spacings) do
        print(string.format("  %d m -> %d blocks", spacing, spacing_counts[spacing]))
    end
end

if expected_spacing then
    print(string.format("Expected spacing: %d m", expected_spacing))
    print(string.format("Blocks not matching expected: %d", spacing_mismatch_blocks))
end

local ok = true
if valid_blocks == 0 then
    ok = false
end
if header_errors > 0 or crc_errors > 0 then
    ok = false
end
if #unique_spacings > 1 then
    ok = false
end
if expected_spacing and spacing_mismatch_blocks > 0 then
    ok = false
end

if ok then
    print("RESULT: PASS")
    os.exit(0)
else
    print("RESULT: FAIL")
    os.exit(1)
end
