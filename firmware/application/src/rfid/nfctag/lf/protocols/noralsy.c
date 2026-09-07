#include "noralsy.h"

#include <stdlib.h>
#include <string.h>

#include "nordic_common.h"
#include "nrf_pwm.h"
#include "protocols.h"
#include "t55xx.h"
#include "tag_base_type.h"
#include "utils/manchester.h"

/*
 * Noralsy LF tag protocol
 * ASK / Manchester, RF/32, 96-bit frame.
 *
 * Frame layout (bit index, MSB-first), from Proxmark3 getnoralsyBits():
 *   bits  0-31 : fixed prefix 0xBB0214FF (12-bit preamble 0xBB0 + rest)
 *   bits 32-43 : card id sub1  (BCD, 12 bits)
 *   bits 44-51 : year          (BCD, 8 bits)
 *   bits 52-55 : 0 (unknown/flag)
 *   bits 56-63 : card id sub2  (BCD, 8 bits)
 *   bits 64-71 : card id sub3  (BCD, 8 bits)
 *   bits 72-75 : checksum1 = chksum(bits+32, 40)
 *   bits 76-79 : checksum2 = chksum(bits, 76)
 *   bits 80-95 : 0
 *
 * id (decimal) is converted to BCD then split:
 *   sub1 = (bcd & 0x0FFF0000) >> 16
 *   sub2 = (bcd & 0x0000FF00) >> 8
 *   sub3 = (bcd & 0x000000FF)
 * checksum = XOR of successive 4-bit nibbles, masked to 4 bits.
 *
 * Slot data (NORALSY_DATA_SIZE bytes):
 *   [0..3] card id   (uint32, big-endian)
 *   [4..5] year      (uint16, big-endian)
 *   [6..7] padding
 *
 * NOTE: a real Noralsy T5577 uses the T5577 Sequence Terminator (ST) to mark
 * the block boundary. This emulator does not emit an explicit ST; it repeats
 * the 96-bit Manchester frame so the reader syncs on the 0xBB0 preamble, the
 * same way Proxmark3's demodNoralsy locates the card. Verify against a real
 * Noralsy reader (or `lf search` on a Proxmark3) before relying on it.
 *
 * Reference: Proxmark3 cmdlfnoralsy.c
 */

#define NORALSY_RAW_SIZE (96)
#define NORALSY_DATA_SIZE (8)
#define NORALSY_T55XX_BLOCK_COUNT (4)  // config + 3 data blocks

// Manchester edge timing for RF/32 (same thresholds as Viking, also RF/32)
#define NORALSY_READ_TIME1_BASE (0x20)
#define NORALSY_READ_TIME2_BASE (0x30)
#define NORALSY_READ_TIME3_BASE (0x40)
#define NORALSY_READ_JITTER_TIME_BASE (0x07)

#define NRF_LOG_MODULE_NAME noralsy_protocol
#include "nrf_log.h"
#include "nrf_log_ctrl.h"
#include "nrf_log_default_backends.h"
NRF_LOG_MODULE_REGISTER();

static const uint8_t noralsy_preamble[12] = {1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0};

static nrf_pwm_values_wave_form_t m_noralsy_pwm_seq_vals[NORALSY_RAW_SIZE] = {};

nrf_pwm_sequence_t const m_noralsy_pwm_seq = {
    .values.p_wave_form = m_noralsy_pwm_seq_vals,
    .length = NRF_PWM_VALUES_LENGTH(m_noralsy_pwm_seq_vals),
    .repeats = 0,
    .end_delay = 0,
};

typedef struct {
    uint8_t data[NORALSY_DATA_SIZE];
    uint8_t bits[NORALSY_RAW_SIZE];  // sliding window of last 96 decoded bits
    uint8_t nbits;
    manchester *modem;
} noralsy_codec;

// ---- helpers ---------------------------------------------------------------

static uint32_t noralsy_dec2bcd(uint32_t dec) {
    uint32_t bcd = 0;
    uint8_t shift = 0;
    while (dec) {
        bcd |= (dec % 10) << (shift * 4);
        dec /= 10;
        shift++;
    }
    return bcd;
}

static uint32_t noralsy_bcd2dec(uint32_t bcd) {
    uint32_t dec = 0;
    uint32_t mul = 1;
    while (bcd) {
        dec += (bcd & 0x0F) * mul;
        mul *= 10;
        bcd >>= 4;
    }
    return dec;
}

// write numbits of value into bits[] MSB-first
static void noralsy_num_to_bits(uint32_t value, uint8_t numbits, uint8_t *bits) {
    for (uint8_t i = 0; i < numbits; i++) {
        bits[i] = (value >> (numbits - 1 - i)) & 0x01;
    }
}

// read numbits from bits[] MSB-first into a value
static uint32_t noralsy_bits_to_num(const uint8_t *bits, uint8_t numbits) {
    uint32_t v = 0;
    for (uint8_t i = 0; i < numbits; i++) {
        v = (v << 1) | (bits[i] & 0x01);
    }
    return v;
}

// XOR of successive 4-bit nibbles, masked to 4 bits (Proxmark noralsy_chksum)
static uint8_t noralsy_chksum(const uint8_t *bits, uint8_t len) {
    uint8_t sum = 0;
    for (uint8_t i = 0; i < len; i += 4) {
        sum ^= (uint8_t)noralsy_bits_to_num(bits + i, 4);
    }
    return sum & 0x0F;
}

// build the 96-bit frame from id (decimal) and year (decimal)
static void noralsy_get_bits(uint32_t id, uint16_t year, uint8_t *bits) {
    memset(bits, 0, NORALSY_RAW_SIZE);

    noralsy_num_to_bits(0xBB0214FF, 32, bits);  // fixed prefix

    uint32_t id_bcd = noralsy_dec2bcd(id);
    uint32_t year_bcd = noralsy_dec2bcd(year) & 0xFF;

    uint16_t sub1 = (id_bcd & 0x0FFF0000) >> 16;
    uint8_t sub2 = (id_bcd & 0x0000FF00) >> 8;
    uint8_t sub3 = (id_bcd & 0x000000FF);

    noralsy_num_to_bits(sub1, 12, bits + 32);
    noralsy_num_to_bits(year_bcd, 8, bits + 44);
    noralsy_num_to_bits(0, 4, bits + 52);
    noralsy_num_to_bits(sub2, 8, bits + 56);
    noralsy_num_to_bits(sub3, 8, bits + 64);

    uint8_t chk1 = noralsy_chksum(bits + 32, 40);
    noralsy_num_to_bits(chk1, 4, bits + 72);
    uint8_t chk2 = noralsy_chksum(bits, 76);
    noralsy_num_to_bits(chk2, 4, bits + 76);
}

// decode the slot data buffer into a 96-bit frame
static void noralsy_frame_from_data(const uint8_t *buf, uint8_t *bits) {
    uint32_t id = ((uint32_t)buf[0] << 24) | ((uint32_t)buf[1] << 16) |
                  ((uint32_t)buf[2] << 8) | (uint32_t)buf[3];
    uint16_t year = ((uint16_t)buf[4] << 8) | (uint16_t)buf[5];
    noralsy_get_bits(id, year, bits);
}

// ---- Manchester read timing (RF/32) ---------------------------------------

static bool noralsy_get_time(uint8_t interval, uint8_t base) {
    return interval >= (base - NORALSY_READ_JITTER_TIME_BASE) &&
           interval <= (base + NORALSY_READ_JITTER_TIME_BASE);
}

static uint8_t noralsy_period(uint8_t interval) {
    if (noralsy_get_time(interval, NORALSY_READ_TIME1_BASE)) {
        return 0;
    }
    if (noralsy_get_time(interval, NORALSY_READ_TIME2_BASE)) {
        return 1;
    }
    if (noralsy_get_time(interval, NORALSY_READ_TIME3_BASE)) {
        return 2;
    }
    return 3;
}

// ---- codec lifecycle -------------------------------------------------------

static noralsy_codec *noralsy_alloc(void) {
    noralsy_codec *codec = malloc(sizeof(noralsy_codec));
    codec->modem = malloc(sizeof(manchester));
    codec->modem->rp = noralsy_period;
    return codec;
}

static void noralsy_free(noralsy_codec *d) {
    if (d->modem) {
        free(d->modem);
        d->modem = NULL;
    }
    free(d);
}

static uint8_t *noralsy_get_data(noralsy_codec *d) {
    return d->data;
}

// ---- decoder (reader mode; not exercised by emulation) ---------------------

static void noralsy_decoder_start(noralsy_codec *d, uint8_t format) {
    memset(d->data, 0, NORALSY_DATA_SIZE);
    memset(d->bits, 0, NORALSY_RAW_SIZE);
    d->nbits = 0;
    manchester_reset(d->modem);
}

static bool noralsy_decode_feed(noralsy_codec *d, bool bit) {
    // sliding window of the last 96 bits
    memmove(d->bits, d->bits + 1, NORALSY_RAW_SIZE - 1);
    d->bits[NORALSY_RAW_SIZE - 1] = bit ? 1 : 0;
    if (d->nbits < NORALSY_RAW_SIZE) {
        d->nbits++;
        return false;
    }

    // preamble at start of window
    if (memcmp(d->bits, noralsy_preamble, sizeof(noralsy_preamble)) != 0) {
        return false;
    }

    // validate checksums
    if (noralsy_chksum(d->bits + 32, 40) != (uint8_t)noralsy_bits_to_num(d->bits + 72, 4)) {
        return false;
    }
    if (noralsy_chksum(d->bits, 76) != (uint8_t)noralsy_bits_to_num(d->bits + 76, 4)) {
        return false;
    }

    // extract id + year
    uint16_t sub1 = noralsy_bits_to_num(d->bits + 32, 12);
    uint8_t year_bcd = noralsy_bits_to_num(d->bits + 44, 8);
    uint8_t sub2 = noralsy_bits_to_num(d->bits + 56, 8);
    uint8_t sub3 = noralsy_bits_to_num(d->bits + 64, 8);

    uint32_t id_bcd = ((uint32_t)sub1 << 16) | ((uint32_t)sub2 << 8) | sub3;
    uint32_t id = noralsy_bcd2dec(id_bcd);
    uint16_t year = noralsy_bcd2dec(year_bcd);

    d->data[0] = (id >> 24) & 0xFF;
    d->data[1] = (id >> 16) & 0xFF;
    d->data[2] = (id >> 8) & 0xFF;
    d->data[3] = id & 0xFF;
    d->data[4] = (year >> 8) & 0xFF;
    d->data[5] = year & 0xFF;
    d->data[6] = 0;
    d->data[7] = 0;
    return true;
}

static bool noralsy_decoder_feed(noralsy_codec *d, uint16_t interval) {
    bool bits[2] = {0};
    int8_t bitlen = 0;
    manchester_feed(d->modem, (uint8_t)interval, bits, &bitlen);
    if (bitlen == -1) {
        d->nbits = 0;
        memset(d->bits, 0, NORALSY_RAW_SIZE);
        return false;
    }
    for (int i = 0; i < bitlen; i++) {
        if (noralsy_decode_feed(d, bits[i])) {
            return true;
        }
    }
    return false;
}

// ---- modulator (emulation) -------------------------------------------------

static const nrf_pwm_sequence_t *noralsy_modulator(noralsy_codec *d, uint8_t *buf) {
    uint8_t bits[NORALSY_RAW_SIZE];
    noralsy_frame_from_data(buf, bits);

    for (int i = 0; i < NORALSY_RAW_SIZE; i++) {
        uint16_t msb = bits[i] ? (1 << 15) : 0x00;
        // counter_top = 32 -> RF/32; value 16 = 50% duty -> Manchester symbol,
        // polarity (msb) selects the transition direction from the data bit.
        m_noralsy_pwm_seq_vals[i].channel_0 = msb | 16;
        m_noralsy_pwm_seq_vals[i].counter_top = 32;
    }
    return &m_noralsy_pwm_seq;
}

const protocol noralsy = {
    .tag_type = TAG_TYPE_NORALSY,
    .data_size = NORALSY_DATA_SIZE,
    .alloc = (codec_alloc)noralsy_alloc,
    .free = (codec_free)noralsy_free,
    .get_data = (codec_get_data)noralsy_get_data,
    .modulator = (modulator)noralsy_modulator,
    .decoder =
        {
            .start = (decoder_start)noralsy_decoder_start,
            .feed = (decoder_feed)noralsy_decoder_feed,
        },
};

uint8_t noralsy_t55xx_writer(uint8_t *data, uint32_t *blks) {
    uint8_t bits[NORALSY_RAW_SIZE];
    noralsy_frame_from_data(data, bits);
    blks[0] = T5577_NORALSY_CONFIG;
    blks[1] = noralsy_bits_to_num(bits, 32);
    blks[2] = noralsy_bits_to_num(bits + 32, 32);
    blks[3] = noralsy_bits_to_num(bits + 64, 32);
    return NORALSY_T55XX_BLOCK_COUNT;
}
