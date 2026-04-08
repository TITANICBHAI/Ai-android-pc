package com.ariaagent.mobile.core.triggers

import android.content.Context
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

/**
 * TriggerStore — persistent storage for scheduled agent triggers.
 *
 * Triggers are stored as a JSON array in SharedPreferences under the key
 * "aria_triggers". Each trigger has a type, goal, app package, time, and
 * enabled flag.
 *
 * Thread safety: all operations are synchronous and must be called from
 * a background coroutine (Dispatchers.IO) in the ViewModel.
 *
 * Trigger types:
 *   TIME_ONCE    — fire once at a specific HH:MM
 *   TIME_DAILY   — fire every day at HH:MM
 *   TIME_WEEKLY  — fire on a specific day of week at HH:MM
 *   APP_LAUNCH   — fire when a specific app is foregrounded (via A11y service)
 *   CHARGING     — fire when the device starts charging
 */

enum class TriggerType(val label: String) {
    TIME_ONCE("Once at time"),
    TIME_DAILY("Daily"),
    TIME_WEEKLY("Weekly"),
    APP_LAUNCH("On app launch"),
    CHARGING("On charging"),
}

data class TriggerItem(
    val id: String             = UUID.randomUUID().toString(),
    val type: TriggerType      = TriggerType.TIME_DAILY,
    val goal: String           = "",
    val goalAppPackage: String = "",
    val watchPackage: String   = "",
    val hourOfDay: Int         = 9,
    val minuteOfHour: Int      = 0,
    val dayOfWeek: Int         = 2,
    val enabled: Boolean       = true,
    val createdAt: Long        = System.currentTimeMillis(),
) {
    val displayTime: String get() = "%02d:%02d".format(hourOfDay, minuteOfHour)

    val dayLabel: String get() = when (dayOfWeek) {
        1 -> "Sunday"; 2 -> "Monday"; 3 -> "Tuesday"; 4 -> "Wednesday"
        5 -> "Thursday"; 6 -> "Friday"; 7 -> "Saturday"; else -> "?"
    }

    fun toJson(): JSONObject = JSONObject().apply {
        put("id",             id)
        put("type",           type.name)
        put("goal",           goal)
        put("goalAppPackage", goalAppPackage)
        put("watchPackage",   watchPackage)
        put("hourOfDay",      hourOfDay)
        put("minuteOfHour",   minuteOfHour)
        put("dayOfWeek",      dayOfWeek)
        put("enabled",        enabled)
        put("createdAt",      createdAt)
    }

    companion object {
        fun fromJson(obj: JSONObject): TriggerItem? = try {
            TriggerItem(
                id             = obj.getString("id"),
                type           = TriggerType.valueOf(obj.getString("type")),
                goal           = obj.optString("goal", ""),
                goalAppPackage = obj.optString("goalAppPackage", ""),
                watchPackage   = obj.optString("watchPackage", ""),
                hourOfDay      = obj.optInt("hourOfDay", 9),
                minuteOfHour   = obj.optInt("minuteOfHour", 0),
                dayOfWeek      = obj.optInt("dayOfWeek", 2),
                enabled        = obj.optBoolean("enabled", true),
                createdAt      = obj.optLong("createdAt", System.currentTimeMillis()),
            )
        } catch (e: Exception) {
            Log.w("TriggerStore", "Failed to parse trigger: ${e.message}")
            null
        }
    }
}

object TriggerStore {

    private const val TAG        = "TriggerStore"
    private const val PREFS_NAME = "aria_triggers_prefs"
    private const val KEY_JSON   = "aria_triggers"

    fun load(context: Context): List<TriggerItem> {
        return try {
            val prefs = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            val raw   = prefs.getString(KEY_JSON, "[]") ?: "[]"
            val arr   = JSONArray(raw)
            (0 until arr.length()).mapNotNull { TriggerItem.fromJson(arr.getJSONObject(it)) }
        } catch (e: Exception) {
            Log.e(TAG, "load failed: ${e.message}")
            emptyList()
        }
    }

    fun save(context: Context, triggers: List<TriggerItem>) {
        try {
            val arr = JSONArray()
            triggers.forEach { arr.put(it.toJson()) }
            context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
                .edit()
                .putString(KEY_JSON, arr.toString())
                .apply()
        } catch (e: Exception) {
            Log.e(TAG, "save failed: ${e.message}")
        }
    }

    fun add(context: Context, trigger: TriggerItem) {
        val current = load(context).toMutableList()
        current.removeAll { it.id == trigger.id }
        current.add(trigger)
        save(context, current)
    }

    fun delete(context: Context, id: String) {
        val current = load(context).filter { it.id != id }
        save(context, current)
    }

    fun toggle(context: Context, id: String): TriggerItem? {
        val current = load(context).toMutableList()
        val idx     = current.indexOfFirst { it.id == id }
        if (idx < 0) return null
        val updated = current[idx].copy(enabled = !current[idx].enabled)
        current[idx] = updated
        save(context, current)
        return updated
    }
}
